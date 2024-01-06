import asyncio
from multiprocessing import Process, Manager
import uvicorn
import sys
import os 
from config import API_SERVER_CONFIG,WORK_CONFIG,CONTROLLER_CONFIG,LOG_PATH

def run_controller(started_event, stoped_event):
    import fastchat.constants
    fastchat.constants.LOGDIR = LOG_PATH
    from fastchat.serve.controller import app, Controller, logger
    
    controller = Controller("shortest_queue")
    sys.modules["fastchat.serve.controller"].controller = controller

    @app.on_event("startup")
    async def app_startup():        
        if started_event:
            started_event.set()
    
    @app.on_event("shutdown")
    async def app_shutdown():        
        if stoped_event:
            stoped_event.set()

    uvicorn.run(app, host=CONTROLLER_CONFIG['host'], port=CONTROLLER_CONFIG['port'], log_level="debug")


def run_openai_api_server():
    import fastchat.constants
    fastchat.constants.LOGDIR = LOG_PATH
    from fastchat.serve.openai_api_server import app, CORSMiddleware, app_settings,logger

    app.add_middleware(
        CORSMiddleware,
        allow_credentials=True,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app_settings.controller_address = f"http://{CONTROLLER_CONFIG['host']}:{CONTROLLER_CONFIG['port']}"
    app_settings.api_keys = API_SERVER_CONFIG['api_keys']
    
    uvicorn.run(app, host=API_SERVER_CONFIG['host'], port=API_SERVER_CONFIG['port'], log_level="debug")

def detect_device():
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except:
        pass
    return "cpu"

def load_embeddings_model(self):
    model_path=WORK_CONFIG['models'][self.model_names[0]]
    if 'bge-' in model_path:
        from langchain.embeddings import HuggingFaceBgeEmbeddings
        if 'zh' in model_path:
            # for chinese model
            query_instruction = "为这个句子生成表示以用于检索相关文章："
        elif 'en' in model_path:
            # for english model
            query_instruction = "Represent this sentence for searching relevant passages:"
        else:
            # maybe ReRanker or else, just use empty string instead
            query_instruction = ""
        embeddings = HuggingFaceBgeEmbeddings(model_name=model_path,
                                                model_kwargs={'device': detect_device()},
                                                query_instruction=query_instruction)
        if "bge-large-zh-noinstruct" in model_path:  # bge large -noinstruct embedding
            embeddings.query_instruction = ""
    else:
        from langchain.embeddings.huggingface import HuggingFaceEmbeddings
        embeddings = HuggingFaceEmbeddings(model_name=model_path,
                                            model_kwargs={'device': detect_device()})
        
    return embeddings

def get_embeddings(self, params):
    if not hasattr(self, "_embeddings_model"):
        self._embeddings_model = load_embeddings_model(self)
    ret = {"embedding": [], "token_num": 0}

    normalized_embeddings = self._embeddings_model.embed_query(params["input"][0])
    ret["token_num"] = len(normalized_embeddings)
    ret["embedding"] = [normalized_embeddings]

    return ret

def run_model_worker(started_event):
    import fastchat.constants
    fastchat.constants.LOGDIR = LOG_PATH
    from fastchat.serve.multi_model_worker import app, worker_id,workers ,worker_map,ModelWorker,GptqConfig,ExllamaConfig,XftConfig,logger
    from fastchat.model.model_adapter import add_model_args
    import argparse

    ModelWorker.get_embeddings=get_embeddings

   

    parser = argparse.ArgumentParser(conflict_handler="resolve")
    parser.add_argument("--host", type=str, default=WORK_CONFIG['host'])
    parser.add_argument("--port", type=int, default=WORK_CONFIG['port'])
    parser.add_argument("--worker-address", type=str, default=f"http://{WORK_CONFIG['host']}:{WORK_CONFIG['port']}")
    parser.add_argument(
        "--controller-address", type=str, default= f"http://{CONTROLLER_CONFIG['host']}:{CONTROLLER_CONFIG['port']}"
    )
    add_model_args(parser)
    # Override the model path to be repeated and align it with model names.
    parser.add_argument(
        "--model-path",
        type=str,
        action="append",
        help="One or more paths to model weights to load. This can be a local folder or a Hugging Face repo ID.",
    )
    parser.add_argument(
        "--model-names",
        type=str,
        action="append",
        help="One or more model names.  Values must be aligned with `--model-path` values.",
    )
    parser.add_argument(
        "--conv-template",
        type=str,
        default=None,
        action="append",
        help="Conversation prompt template. Values must be aligned with `--model-path` values. If only one value is provided, it will be repeated for all models.",
    )
    parser.add_argument("--limit-worker-concurrency", type=int, default=5)
    parser.add_argument("--stream-interval", type=int, default=2)
    parser.add_argument("--no-register", action="store_true")
    parser.add_argument(
        "--ssl",
        action="store_true",
        required=False,
        default=False,
        help="Enable SSL. Requires OS Environment variables 'SSL_KEYFILE' and 'SSL_CERTFILE'.",
    )
    args = parser.parse_args()


    if args.gpus:
        if len(args.gpus.split(",")) < args.num_gpus:
            raise ValueError(
                f"Larger --num-gpus ({args.num_gpus}) than --gpus {args.gpus}!"
            )
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    gptq_config = GptqConfig(
        ckpt=args.gptq_ckpt or args.model_path,
        wbits=args.gptq_wbits,
        groupsize=args.gptq_groupsize,
        act_order=args.gptq_act_order,
    )
    if args.enable_exllama:
        exllama_config = ExllamaConfig(
            max_seq_len=args.exllama_max_seq_len,
            gpu_split=args.exllama_gpu_split,
            cache_8bit=args.exllama_cache_8bit,
        )
    else:
        exllama_config = None
    if args.enable_xft:
        xft_config = XftConfig(
            max_seq_len=args.xft_max_seq_len,
            data_type=args.xft_dtype,
        )
        if args.device != "cpu":
            print("xFasterTransformer now is only support CPUs. Reset device to CPU")
            args.device = "cpu"
    else:
        xft_config = None
    
    # Override
    args.device = detect_device()
    
    # Launch all workers
    workers = []
    for key in WORK_CONFIG['models'].keys():
        model_names=[key]
        model_path=WORK_CONFIG['models'][key]
        w = ModelWorker(
            args.controller_address,
            args.worker_address,
            worker_id,
            model_path,
            model_names,
            args.limit_worker_concurrency,
            args.no_register,
            device=args.device,
            num_gpus=args.num_gpus,
            max_gpu_memory=args.max_gpu_memory,
            load_8bit=args.load_8bit,
            cpu_offloading=args.cpu_offloading,
            gptq_config=gptq_config,
            exllama_config=exllama_config,
            xft_config=xft_config,
            stream_interval=args.stream_interval,
            conv_template=args.conv_template,
        )
        workers.append(w)
        for model_name in model_names:
            worker_map[model_name] = w

    # Register all models
    model_names=[]
    for key in WORK_CONFIG['models'].keys():
        model_names.append(key)
    url = args.controller_address + "/register_worker"
    data = {
        "worker_name": workers[0].worker_addr,
        "check_heart_beat": not args.no_register,
        "worker_status": {
            "model_names": model_names,
            "speed": 1,
            "queue_length": sum([w.get_queue_length() for w in workers]),
        },
    }
    import requests
    r = requests.post(url, json=data)
    assert r.status_code == 200
    
    sys.modules["fastchat.serve.multi_model_worker"].workers = workers 
    sys.modules["fastchat.serve.multi_model_worker"].worker_map = worker_map  

    @app.on_event("startup")
    async def app_startup():        
        if started_event:
            started_event.set()
    uvicorn.run(app, host=args.host, port=args.port, log_level="debug")

def start_main_server():
    import sys
    import signal

    def handler(signalname):
        """
        Python 3.9 has `signal.strsignal(signalnum)` so this closure would not be needed.
        Also, 3.8 includes `signal.valid_signals()` that can be used to create a mapping for the same purpose.
        """
        def f(signal_received, frame):
            raise KeyboardInterrupt(f"{signalname} received")
        return f

    # This will be inherited by the child process if it is forked (not spawned)
    signal.signal(signal.SIGINT, handler("SIGINT"))
    signal.signal(signal.SIGTERM, handler("SIGTERM"))
    manager = Manager()
    controller_started = manager.Event()
    controller_stoped = manager.Event()
    worker_started = manager.Event()
    process_list=[]
    process_list.append(Process(
                target=run_controller,
                kwargs=dict(started_event=controller_started,stoped_event = controller_stoped),
                name=f"controller",
                daemon=True,
            ))
    process_list.append(Process(
                target=run_openai_api_server,
                name=f"openai_api_server",
                daemon=True,
            ))    
    process_list.append(Process(
                target=run_model_worker,      
                kwargs=dict(started_event=worker_started),          
                name=f"model_worker",
                daemon=True,
            ))    
    for p in process_list:          
        p.start()
        if(p.name=="controller"):
            controller_started.wait() # 等待controller启动完成

    worker_started.wait()

    print(f"Local-LLM-Server is successfully started, please use http://{API_SERVER_CONFIG['host']}:{API_SERVER_CONFIG['port']} to access the OpenAI-compatible interfaces")
    print(f"Local-LLM-Server 启动成功，请使用 http://{API_SERVER_CONFIG['host']}:{API_SERVER_CONFIG['port']} 访问 OpenAI 接口")
 
    controller_stoped.wait()
    for p in process_list:
        p.kill()


if __name__ == '__main__':
    start_main_server()

    
