using ChatAIze.SemanticIndex;

var database = new SemanticDatabase<string>();

if (!File.Exists("test-database.json"))
{
    await database.AddAsync("cat");
    await database.AddAsync("dog");
    await database.AddAsync("fish");

    await database.AddAsync("apple");
    await database.AddAsync("banana");
    await database.AddAsync("orange");

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
