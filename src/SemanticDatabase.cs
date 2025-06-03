using System.Numerics.Tensors;
using System.Text.Json;
using ChatAIze.GenerativeCS.Clients;
using ChatAIze.GenerativeCS.Constants;
using ChatAIze.GenerativeCS.Options.OpenAI;

namespace ChatAIze.SemanticIndex;

/// <summary>
/// Provides a simple in-memory vector database built on top of
/// <see cref="OpenAIClient"/> embeddings.
/// </summary>
/// <typeparam name="T">The type of items stored in the database.</typeparam>
public class SemanticDatabase<T>
{
    /// <summary>
    /// Lock used to synchronize access to the record list.
    /// </summary>
    protected readonly ReaderWriterLockSlim _lock = new();

    /// <summary>
    /// Client used to retrieve embeddings for stored items.
    /// </summary>
    protected readonly OpenAIClient _client = new();

    /// <summary>
    /// Options used when requesting embeddings.
    /// </summary>
    protected readonly EmbeddingOptions _embeddingOptions = new()
    {
        Model = EmbeddingModels.OpenAI.TextEmbedding3Large
    };

    /// <summary>
    /// Internal list of semantic records.
    /// </summary>
    protected List<SemanticRecord<T>> _records = [];

    /// <summary>
    /// Initializes a new instance of the <see cref="SemanticDatabase{T}"/> class.
    /// </summary>
    public SemanticDatabase() { }

    /// <summary>
    /// Initializes a new instance of the <see cref="SemanticDatabase{T}"/> class
    /// using the specified OpenAI API key.
    /// </summary>
    /// <param name="apiKey">The OpenAI API key for the <see cref="OpenAIClient"/>.</param>
    public SemanticDatabase(string apiKey)
    {
        _client = new OpenAIClient(apiKey);
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="SemanticDatabase{T}"/> class
    /// using the provided <see cref="OpenAIClient"/> instance.
    /// </summary>
    /// <param name="client">The client used for generating embeddings.</param>
    public SemanticDatabase(OpenAIClient client)
    {
        _client = client;
    }

    /// <summary>
    /// Gets or sets the OpenAI API key used by the underlying client.
    /// </summary>
    public string? ApiKey
    {
        get => _client.ApiKey;
        set => _client.ApiKey = value;
    }

    /// <summary>
    /// Gets or sets the embedding model to use when generating embeddings.
    /// </summary>
    public string EmbeddingModel
    {
        get => _embeddingOptions.Model;
        set => _embeddingOptions.Model = value;
    }

    /// <summary>
    /// Gets or sets the duplicate handling strategy.
    /// </summary>
    public DuplicateHandling DuplicateHandling { get; set; } = DuplicateHandling.Update;

    /// <summary>
    /// Gets a read-only view of all records stored in the database.
    /// </summary>
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

    /// <summary>
    /// Gets the number of records in the database.
    /// </summary>
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

    /// <summary>
    /// Loads a semantic database from a file.
    /// </summary>
    /// <param name="filePath">The path to the JSON file.</param>
    /// <param name="apiKey">Optional OpenAI API key used when creating the database.</param>
    /// <param name="cancellationToken">Token used to cancel the operation.</param>
    /// <returns>A new <see cref="SemanticDatabase{T}"/> populated from the file.</returns>
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

    /// <summary>
    /// Adds a single item to the database.
    /// </summary>
    /// <param name="item">The item to add.</param>
    /// <param name="cancellationToken">Token used to cancel the operation.</param>
    public virtual async Task AddAsync(T item, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(item, nameof(item));

        var json = JsonSerializer.Serialize(item);
        var embedding = await _client.GetEmbeddingAsync(json, _embeddingOptions, cancellationToken: cancellationToken);
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

    /// <summary>
    /// Adds a collection of items to the database.
    /// </summary>
    /// <param name="items">The items to add.</param>
    /// <param name="cancellationToken">Token used to cancel the operation.</param>
    public async Task AddRangeAsync(IEnumerable<T> items, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(items, nameof(items));

        foreach (var item in items)
        {
            await AddAsync(item, cancellationToken);
        }
    }

    /// <summary>
    /// Adds items from an asynchronous stream to the database.
    /// </summary>
    /// <param name="items">The asynchronous stream of items.</param>
    /// <param name="cancellationToken">Token used to cancel the operation.</param>
    public async Task AddRangeAsync(IAsyncEnumerable<T> items, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(items, nameof(items));

        await foreach (var item in items.WithCancellation(cancellationToken))
        {
            await AddAsync(item, cancellationToken);
        }
    }

    /// <summary>
    /// Determines whether the specified item exists in the database.
    /// </summary>
    /// <param name="item">The item to check for.</param>
    /// <returns><see langword="true"/> if the item is present; otherwise <see langword="false"/>.</returns>
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

    /// <summary>
    /// Gets a list of all items stored in the database.
    /// </summary>
    /// <returns>A list containing all stored items.</returns>
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

    /// <summary>
    /// Searches the database for items similar to the given embedding.
    /// </summary>
    /// <param name="embedding">The embedding to search for.</param>
    /// <param name="count">The maximum number of results to return.</param>
    /// <returns>An enumerable of the most similar items.</returns>
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

    /// <summary>
    /// Returns the single most similar item to the specified embedding.
    /// </summary>
    /// <param name="embedding">The embedding to search for.</param>
    /// <returns>The most similar item or <see langword="null"/> if none exist.</returns>
    public T? SearchFirst(float[] embedding)
    {
        ArgumentNullException.ThrowIfNull(embedding, nameof(embedding));

        var results = Search(embedding, 1);
        return results.FirstOrDefault();
    }

    /// <summary>
    /// Searches the database asynchronously using a text query.
    /// </summary>
    /// <param name="query">The text to search for.</param>
    /// <param name="count">The maximum number of results to return.</param>
    /// <param name="cancellationToken">Token used to cancel the operation.</param>
    /// <returns>An enumerable of matching items.</returns>
    public virtual async Task<IEnumerable<T>> SearchAsync(string query, int count = 10, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(query, nameof(query));

        var embedding = await _client.GetEmbeddingAsync(query, cancellationToken: cancellationToken);
        return Search(embedding, count);
    }

    /// <summary>
    /// Returns the single most similar item to the specified text query.
    /// </summary>
    /// <param name="query">The text to search for.</param>
    /// <param name="cancellationToken">Token used to cancel the operation.</param>
    /// <returns>The most similar item or <see langword="null"/> if none exist.</returns>
    public virtual async Task<T?> SearchFirstAsync(string query, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(query, nameof(query));

        var results = await SearchAsync(query, 1, cancellationToken);
        return results.FirstOrDefault();
    }

    /// <summary>
    /// Searches the database asynchronously using an arbitrary object as the query.
    /// </summary>
    /// <param name="query">The object to search for.</param>
    /// <param name="count">The maximum number of results to return.</param>
    /// <param name="cancellationToken">Token used to cancel the operation.</param>
    /// <returns>An enumerable of matching items.</returns>
    public virtual async Task<IEnumerable<T>> SearchAsync(object query, int count = 10, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(query, nameof(query));

        var json = JsonSerializer.Serialize(query);
        return await SearchAsync(json, count, cancellationToken);
    }

    /// <summary>
    /// Returns the single most similar item to the specified object query.
    /// </summary>
    /// <param name="query">The object to search for.</param>
    /// <param name="cancellationToken">Token used to cancel the operation.</param>
    /// <returns>The most similar item or <see langword="null"/> if none exist.</returns>
    public virtual async Task<T?> SearchFirstAsync(object query, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(query, nameof(query));

        var results = await SearchAsync(query, 1, cancellationToken);
        return results.FirstOrDefault();
    }

    /// <summary>
    /// Regenerates embeddings for all records in the database.
    /// </summary>
    /// <param name="cancellationToken">Token used to cancel the operation.</param>
    public async Task RefreshEmbeddingsAsync(CancellationToken cancellationToken = default)
    {
        // Take a snapshot of the items so we don't hold the lock during the network calls
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

        foreach (var record in records)
        {
            var json = JsonSerializer.Serialize(record.Item);
            record.Embedding = await _client.GetEmbeddingAsync(json, _embeddingOptions, cancellationToken: cancellationToken);
        }

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

    /// <summary>
    /// Removes the specified item from the database.
    /// </summary>
    /// <param name="item">The item to remove.</param>
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

    /// <summary>
    /// Removes a collection of items from the database.
    /// </summary>
    /// <param name="items">The items to remove.</param>
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

    /// <summary>
    /// Removes all records from the database.
    /// </summary>
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

    /// <summary>
    /// Loads database records from a JSON file.
    /// </summary>
    /// <param name="filePath">The path to the file.</param>
    /// <param name="cancellationToken">Token used to cancel the operation.</param>
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

    /// <summary>
    /// Saves the database to a JSON file.
    /// </summary>
    /// <param name="filePath">The file path to save to.</param>
    /// <param name="cancellationToken">Token used to cancel the operation.</param>
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
