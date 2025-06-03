using System.Numerics.Tensors;
using System.Text.Json;
using ChatAIze.GenerativeCS.Clients;

namespace ChatAIze.SemanticIndex;

public class SemanticDatabase<T>
{
    protected readonly ReaderWriterLockSlim _lock = new();

    protected readonly OpenAIClient _client = new();

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

    public virtual async Task AddAsync(T item, CancellationToken cancellationToken = default)
    {
        var json = JsonSerializer.Serialize(item);
        var embedding = await _client.GetEmbeddingAsync(json, cancellationToken: cancellationToken);
        var record = new SemanticRecord<T>(item, embedding);

        _lock.EnterWriteLock();

        try
        {
            _records.Add(record);
        }
        finally
        {
            _lock.ExitWriteLock();
        }
    }

    public virtual IEnumerable<T> Search(float[] embedding, int count = 10)
    {
        var results = new SortedList<float, T>();
        _lock.EnterReadLock();

        try
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
        finally
        {
            _lock.ExitReadLock();
        }

        return results.Values.Reverse();
    }

    public virtual async Task<IEnumerable<T>> SearchAsync(string query, int count = 10, CancellationToken cancellationToken = default)
    {
        var embedding = await _client.GetEmbeddingAsync(query, cancellationToken: cancellationToken);
        return Search(embedding, count);
    }

    public virtual async Task<IEnumerable<T>> SearchAsync(object query, int count = 10, CancellationToken cancellationToken = default)
    {
        var json = JsonSerializer.Serialize(query);
        return await SearchAsync(json, count, cancellationToken);
    }

    public virtual void Remove(T item)
    {
        _lock.EnterWriteLock();

        try
        {
            _records.RemoveAll(r => r.Item!.Equals(item));
        }
        finally
        {
            _lock.ExitWriteLock();
        }
    }

    public virtual async Task LoadAsync(string filePath, CancellationToken cancellationToken = default)
    {
        using var stream = File.OpenRead(filePath);
        var records = await JsonSerializer.DeserializeAsync<List<SemanticRecord<T>>>(stream, cancellationToken: cancellationToken) ?? [];

        _lock.EnterWriteLock();

        try
        {
            _records = records;
        }
        finally
        {
            _lock.ExitWriteLock();
        }
    }

    public virtual async Task SaveAsync(string filePath, CancellationToken cancellationToken = default)
    {
        List<SemanticRecord<T>> records;
        _lock.EnterReadLock();

        try
        {
            records = [.. _records];
        }
        finally
        {
            _lock.ExitReadLock();
        }

        using var stream = File.Create(filePath);
        await JsonSerializer.SerializeAsync(stream, records, cancellationToken: cancellationToken);
    }
}
