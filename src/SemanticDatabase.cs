using System.Numerics.Tensors;
using System.Text;
using ChatAIze.GenerativeCS.Clients;
using ChatAIze.GenerativeCS.Options.OpenAI;

namespace ChatAIze.SemanticIndex;

public sealed class SemanticDatabase
{
    private readonly OpenAIClient _client = new();

    public required string Path { get; set; }

    public required string ApiKey { get; set; }

    public async Task AddAsync(string item, ICollection<string>? tags = null, CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(item);

        var embedding = await GetEmbeddingAsync(item, cancellationToken);
        var builder = new StringBuilder();

        builder.AppendJoin(',', tags ?? Enumerable.Empty<string>());
        builder.Append(';');
        builder.AppendJoin(',', embedding);
        builder.Append(';');
        builder.Append(item);

        using var writer = new StreamWriter(Path, append: true);
        await writer.WriteLineAsync(builder.ToString());
    }

    public async Task<IList<string>> FindAsync(string query, ICollection<string>? tags = null, int count = 10, CancellationToken cancellationToken = default)
    {
        var queryEmbedding = await GetEmbeddingAsync(query, cancellationToken);
        var results = new SortedList<float, string>();

        using var reader = new StreamReader(Path);

        while (!reader.EndOfStream)
        {
            var line = await reader.ReadLineAsync(cancellationToken);
            if (line is null)
            {
                break;
            }

            var parts = line.Split(';');
            if (parts.Length != 3 || tags is not null && tags.Any(t => !parts[0].Contains(t)))
            {
                continue;
            }

            var itemEmbedding = parts[1].Split(',').Select(float.Parse).ToArray();
            var similarity = TensorPrimitives.CosineSimilarity(queryEmbedding, itemEmbedding);

            while (results.ContainsKey(similarity))
            {
                similarity += float.Epsilon;
            }

            if (results.Count < count)
            {
                results.Add(similarity, parts[2]);
            }
            else if (similarity > results.Keys[^1])
            {
                results.RemoveAt(results.Count - 1);
                results.Add(similarity, parts[2]);
            }
        }

        return results.Values;
    }

    public Task RemoveAsync(string item, ICollection<string> tags, CancellationToken cancellationToken = default)
    {
        throw new NotImplementedException();
    }

    public Task RemoveAsync(ICollection<string> tags, CancellationToken cancellationToken = default)
    {
        throw new NotImplementedException();
    }

    private async Task<float[]> GetEmbeddingAsync(string item, CancellationToken cancellationToken = default)
    {
        var options = new EmbeddingOptions(apiKey: ApiKey);
        var embedding = await _client.GetEmbeddingAsync(item, options, cancellationToken: cancellationToken);

        return embedding;
    }
}
