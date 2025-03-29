using ChatAIze.SemanticIndex;

var database = new SemanticDatabase<string>();

if (!File.Exists("test-database.json"))
{
    var tasks = new List<Task>
    {
        database.AddAsync("cat"),
        database.AddAsync("dog"),
        database.AddAsync("fish"),
        database.AddAsync("apple"),
        database.AddAsync("banana"),
        database.AddAsync("orange")
    };

    await Task.WhenAll(tasks);
    await database.SaveAsync("test-database.json");
}
else
{
    await database.LoadAsync("test-database.json");
}

var animals = await database.SearchAsync("animal", 3);
foreach (var animal in animals)
{
    Console.WriteLine(animal);
}
