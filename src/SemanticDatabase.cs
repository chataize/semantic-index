using System.Numerics.Tensors;
using System.Text.Json;
using ChatAIze.GenerativeCS.Clients;

namespace ChatAIze.SemanticIndex;

public sealed class SemanticDatabase<T>
{
    private readonly Lock _lock = new();

    private readonly OpenAIClient _client = new();

    private List<SemanticRecord<T>> _records = [];

    public SemanticDatabase() { }

    public SemanticDatabase(string apiKey)
    {
        _client = new OpenAIClient(apiKey);
    }

    public SemanticDatabase(OpenAIClient client)
    {
        _client = client;
    }

    public async Task AddAsync(T item, CancellationToken cancellationToken = default)
    {
        var json = JsonSerializer.Serialize(item);
        var embedding = await _client.GetEmbeddingAsync(json, cancellationToken: cancellationToken);
        var record = new SemanticRecord<T>(item, embedding);

        lock (_lock)
        {
            _records.Add(record);
        }
    }

    public IEnumerable<T> Search(float[] embedding, int count = 10)
    {
        var results = new SortedList<float, T>();

        lock (_lock)
        {
            foreach (var record in _records)
            {
                var similarity = TensorPrimitives.Dot(record.Embedding, embedding);

                if (results.Count < count)
                {
                    while (results.ContainsKey(similarity))
                    {
                        similarity += 1e-6f;
                    }

                    results.Add(similarity, record.Item);
                }
                else if (similarity > results.Keys[0])
                {
                    while (results.ContainsKey(similarity))
                    {
                        similarity += 1e-6f;
                    }

                    results.RemoveAt(0);
                    results.Add(similarity, record.Item);
                }
            }
        }

        return results.Values.Reverse();
    }

    public async Task<IEnumerable<T>> SearchAsync(string query, int count = 10, CancellationToken cancellationToken = default)
    {
        var embedding = await _client.GetEmbeddingAsync(query, cancellationToken: cancellationToken);
        return Search(embedding, count);
    }

    public async Task<IEnumerable<T>> SearchAsync(object query, int count = 10, CancellationToken cancellationToken = default)
    {
        var json = JsonSerializer.Serialize(query);
        return await SearchAsync(json, count, cancellationToken);
    }

    public void Remove(T item)
    {
        lock (_lock)
        {
            _records.RemoveAll(r => r.Item!.Equals(item));
        }
    }

    public async Task LoadAsync(string filePath, CancellationToken cancellationToken = default)
    {
        using var stream = File.OpenRead(filePath);
        var records = await JsonSerializer.DeserializeAsync<List<SemanticRecord<T>>>(stream, cancellationToken: cancellationToken) ?? [];

        lock (_lock)
        {
            _records = records;
        }
    }

    public async Task SaveAsync(string filePath, CancellationToken cancellationToken = default)
    {
        List<SemanticRecord<T>> records;
        lock (_lock)
        {
            records = [.. _records];
        }

        using var stream = File.Create(filePath);
        await JsonSerializer.SerializeAsync(stream, records, cancellationToken: cancellationToken);
    }
}
