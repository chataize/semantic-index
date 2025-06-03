using System.Numerics.Tensors;
using System.Text.Json;
using ChatAIze.GenerativeCS.Clients;

namespace ChatAIze.SemanticIndex;

public class SemanticDatabase<T>
{
    protected readonly ReaderWriterLockSlim _lock = new();

    protected readonly OpenAIClient _client = new();

    protected List<SemanticRecord<T>> _records = [];

    public SemanticDatabase() { }

    public SemanticDatabase(string apiKey)
    {
        _client = new OpenAIClient(apiKey);
    }

    public SemanticDatabase(OpenAIClient client)
    {
        _client = client;
    }

    public string? ApiKey
    {
        get => _client.ApiKey;
        set => _client.ApiKey = value;
    }

    public DuplicateHandling DuplicateHandling { get; set; } = DuplicateHandling.Update;

    public IReadOnlyList<SemanticRecord<T>> Records
    {
        get
        {
            _lock.EnterReadLock();

            try
            {
                return _records.AsReadOnly();
            }
            finally
            {
                _lock.ExitReadLock();
            }
        }
    }

    public int Count
    {
        get
        {
            _lock.EnterReadLock();

            try
            {
                return _records.Count;
            }
            finally
            {
                _lock.ExitReadLock();
            }
        }
    }

    public static async Task<SemanticDatabase<T>> FromFileAsync(string filePath, string? apiKey = null, CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(filePath, nameof(filePath));

        SemanticDatabase<T> database;
        if (apiKey is not null)
        {
            database = new SemanticDatabase<T>(apiKey);
        }
        else
        {
            database = new SemanticDatabase<T>();
        }

        await database.LoadAsync(filePath, cancellationToken);
        return database;
    }

    public virtual async Task AddAsync(T item, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(item, nameof(item));

        var json = JsonSerializer.Serialize(item);
        var embedding = await _client.GetEmbeddingAsync(json, cancellationToken: cancellationToken);
        var record = new SemanticRecord<T>(item, embedding);

        _lock.EnterWriteLock();

        try
        {
            if (DuplicateHandling != DuplicateHandling.Allow && _records.Any(r => r.Item!.Equals(item)))
            {
                if (DuplicateHandling == DuplicateHandling.Update)
                {
                    _records.RemoveAll(r => r.Item!.Equals(item));
                }
                else if (DuplicateHandling == DuplicateHandling.Skip)
                {
                    return;
                }
                else if (DuplicateHandling == DuplicateHandling.Throw)
                {
                    throw new InvalidOperationException("Item already exists in the database.");
                }
            }

            _records.Add(record);
        }
        finally
        {
            _lock.ExitWriteLock();
        }
    }

    public async Task AddRangeAsync(IEnumerable<T> items, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(items, nameof(items));

        foreach (var item in items)
        {
            await AddAsync(item, cancellationToken);
        }
    }

    public async Task AddRangeAsync(IAsyncEnumerable<T> items, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(items, nameof(items));

        await foreach (var item in items.WithCancellation(cancellationToken))
        {
            await AddAsync(item, cancellationToken);
        }
    }

    public bool Contains(T item)
    {
        ArgumentNullException.ThrowIfNull(item, nameof(item));

        _lock.EnterReadLock();

        try
        {
            return _records.Any(r => r.Item!.Equals(item));
        }
        finally
        {
            _lock.ExitReadLock();
        }
    }

    public List<T> GetAll()
    {
        _lock.EnterReadLock();

        try
        {
            return [.. _records.Select(r => r.Item)];
        }
        finally
        {
            _lock.ExitReadLock();
        }
    }

    public virtual IEnumerable<T> Search(float[] embedding, int count = 10)
    {
        ArgumentNullException.ThrowIfNull(embedding, nameof(embedding));

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

    public T? SearchFirst(float[] embedding)
    {
        ArgumentNullException.ThrowIfNull(embedding, nameof(embedding));

        var results = Search(embedding, 1);
        return results.FirstOrDefault();
    }

    public virtual async Task<IEnumerable<T>> SearchAsync(string query, int count = 10, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(query, nameof(query));

        var embedding = await _client.GetEmbeddingAsync(query, cancellationToken: cancellationToken);
        return Search(embedding, count);
    }

    public virtual async Task<T?> SearchFirstAsync(string query, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(query, nameof(query));

        var results = await SearchAsync(query, 1, cancellationToken);
        return results.FirstOrDefault();
    }

    public virtual async Task<IEnumerable<T>> SearchAsync(object query, int count = 10, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(query, nameof(query));

        var json = JsonSerializer.Serialize(query);
        return await SearchAsync(json, count, cancellationToken);
    }

    public virtual async Task<T?> SearchFirstAsync(object query, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(query, nameof(query));

        var results = await SearchAsync(query, 1, cancellationToken);
        return results.FirstOrDefault();
    }

    public async Task RefreshEmbeddingsAsync(CancellationToken cancellationToken = default)
    {
        _lock.EnterWriteLock();

        try
        {
            foreach (var record in _records)
            {
                var json = JsonSerializer.Serialize(record.Item);
                record.Embedding = await _client.GetEmbeddingAsync(json, cancellationToken: cancellationToken);
            }
        }
        finally
        {
            _lock.ExitWriteLock();
        }
    }

    public virtual void Remove(T item)
    {
        ArgumentNullException.ThrowIfNull(item, nameof(item));

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

    public void RemoveRange(IEnumerable<T> items)
    {
        ArgumentNullException.ThrowIfNull(items, nameof(items));

        _lock.EnterWriteLock();

        try
        {
            foreach (var item in items)
            {
                _records.RemoveAll(r => r.Item!.Equals(item));
            }
        }
        finally
        {
            _lock.ExitWriteLock();
        }
    }

    public virtual void Clear()
    {
        _lock.EnterWriteLock();

        try
        {
            _records.Clear();
        }
        finally
        {
            _lock.ExitWriteLock();
        }
    }

    public virtual async Task LoadAsync(string filePath, CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(filePath, nameof(filePath));

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
        ArgumentException.ThrowIfNullOrWhiteSpace(filePath, nameof(filePath));

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
