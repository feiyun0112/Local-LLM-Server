using Microsoft.SemanticKernel;
using System.Text;

Console.OutputEncoding = Encoding.Unicode;

var kernel = Kernel.CreateBuilder()
        .AddOpenAIChatCompletion(
             modelId: "ChatModel",
             apiKey: "NoKey",
             httpClient: new HttpClient(new MyHandler())
        ).Build();

var prompt = "请自我介绍一下?";
var result = await kernel.InvokePromptAsync(prompt);
var answer = result.GetValue<string>();
Console.WriteLine(answer);

//由于 Microsoft.SemanticKernel 没提供直接设置 OpenAI 服务器地址的方法，
//所以需要自定义一个 DelegatingHandler，将 OpenAI 服务器地址修改为 Local-LLM-Server 地址。
class MyHandler : DelegatingHandler
{
    public MyHandler()
        : base(new HttpClientHandler())
    {
    }
    protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
    {
        var newUriBuilder = new UriBuilder(request.RequestUri);
        newUriBuilder.Scheme = "http";
        newUriBuilder.Host = "127.0.0.1";
        newUriBuilder.Port = 21000;

        request.RequestUri = newUriBuilder.Uri;
        return base.SendAsync(request, cancellationToken);
    }
}