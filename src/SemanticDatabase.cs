using System.Numerics.Tensors;
using System.Text.Json;
using ChatAIze.GenerativeCS.Clients;

namespace ChatAIze.SemanticIndex;

public sealed class SemanticDatabase<T>
{
    private readonly OpenAIClient _client = new();

    private List<SemanticRecord<T>> _records = [];

    public async Task AddAsync(T item)
    {
        var json = JsonSerializer.Serialize(item);
        var embedding = await _client.GetEmbeddingAsync(json);
        var record = new SemanticRecord<T>(item, embedding);

        _records.Add(record);
    }

    public IEnumerable<T> Search(float[] embedding, int count = 10)
    {
        var results = new SortedList<float, T>();

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

        return results.Values.Reverse();
    }

    public async Task<IEnumerable<T>> SearchAsync(string query, int count = 10)
    {
        var embedding = await _client.GetEmbeddingAsync(query);
        return await SearchAsync(embedding, count);
    }

    public async Task<IEnumerable<T>> SearchAsync(object query, int count = 10)
    {
        var json = JsonSerializer.Serialize(query);
        return await SearchAsync(json, count);
    }

    public void Remove(T item)
    {
        _records.RemoveAll(r => r.Item!.Equals(item));
    }

    public async Task LoadAsync(string filePath)
    {
        using var stream = File.OpenRead(filePath);
        _records = await JsonSerializer.DeserializeAsync<List<SemanticRecord<T>>>(stream) ?? [];
    }

    public async Task SaveAsync(string filePath)
    {
        using var stream = File.Create(filePath);
        await JsonSerializer.SerializeAsync(stream, _records);
    }
}
