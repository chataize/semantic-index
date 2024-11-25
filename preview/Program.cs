using ChatAIze.SemanticIndex;

var apiKey = Environment.GetEnvironmentVariable("OPENAI_API_KEY")!;
var database = new SemanticDatabase("app.db", apiKey);

//await database.AddAsync("cat", ["animal"]);
//await database.AddAsync("dog", ["animal"]);
//await database.AddAsync("apple", ["food"]);
//await database.AddAsync("banana", ["food"]);

var results = await database.FindAsync("wolf", tags: ["animal"], count: 2);
foreach (var result in results)
{
    Console.WriteLine(result);
}
