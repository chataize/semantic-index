using System.Numerics.Tensors;
using System.Text;
using ChatAIze.GenerativeCS.Clients;
using ChatAIze.GenerativeCS.Constants;
using ChatAIze.GenerativeCS.Options.OpenAI;

namespace ChatAIze.SemanticIndex;

public sealed class SemanticDatabase(string path, string apiKey, string model = EmbeddingModels.OpenAI.TextEmbedding3Small, int vectorSize = 1536)
{
    private const char PropertySeparator = '\u241D';

    private const char TagSeparator = '\u241F';

    private readonly OpenAIClient _client = new();

    private readonly EmbeddingOptions _options = new(model: model, apiKey: apiKey);

    public async Task AddAsync(string item, ICollection<string>? tags = null, CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(item);

        var embedding = await GetEmbeddingAsync(item, cancellationToken);
        var magnitude = Math.Sqrt(TensorPrimitives.SumOfSquares(embedding));
        var builder = new StringBuilder();

        builder.AppendJoin(TagSeparator, tags ?? Enumerable.Empty<string>());
        builder.Append(PropertySeparator);
        builder.AppendJoin(TagSeparator, embedding);
        builder.Append(PropertySeparator);
        builder.Append(magnitude);
        builder.Append(PropertySeparator);
        builder.Append(item);

        using var writer = new StreamWriter(path, append: true);
        await writer.WriteLineAsync(builder.ToString());
    }

    public async Task<IList<string>> FindAsync(string query, ICollection<string>? tags = null, int count = 10, CancellationToken cancellationToken = default)
    {
        var queryEmbedding = await GetEmbeddingAsync(query, cancellationToken);
        var queryMagnitude = Math.Sqrt(TensorPrimitives.SumOfSquares(queryEmbedding));
        var results = new SortedList<double, string>();

        using var reader = new StreamReader(path);

        while (!reader.EndOfStream)
        {
            var line = await reader.ReadLineAsync(cancellationToken);
            if (line is null)
            {
                break;
            }

            var parts = line.Split(PropertySeparator);
            if (parts.Length != 3 || tags is not null && !HasAllTags(parts[0], tags))
            {
                continue;
            }

            var itemEmbedding = ParseFloatArray(parts[1]);
            var itemMagnitude = float.Parse(parts[2]);
            var similarity = TensorPrimitives.Dot(queryEmbedding, itemEmbedding) / (queryMagnitude * itemMagnitude);

            while (results.ContainsKey(similarity))
            {
                similarity += double.Epsilon;
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

    private static bool HasAllTags(ReadOnlySpan<char> tags, ICollection<string> requiredTags)
    {
        foreach (var requiredTag in requiredTags)
        {
            var found = false;

            foreach (var tag in tags.Split(TagSeparator))
            {
                if (tags[tag] == requiredTag)
                {
                    found = true;
                    break;
                }
            }

            if (!found)
            {
                return false;
            }
        }

        return true;
    }

    private async Task<float[]> GetEmbeddingAsync(string item, CancellationToken cancellationToken = default)
    {
        var embedding = await _client.GetEmbeddingAsync(item, _options, cancellationToken: cancellationToken);
        return embedding;
    }

    private float[] ParseFloatArray(ReadOnlySpan<char> embedding)
    {
        var parts = embedding.Split(TagSeparator);
        var result = new float[vectorSize];
        var index = 0;

        foreach (var part in parts)
        {
            result[index++] = float.Parse(embedding[part]);
        }

        return result;
    }
}
