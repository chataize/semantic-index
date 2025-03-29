using System.Diagnostics.CodeAnalysis;

namespace ChatAIze.SemanticIndex;

internal sealed record SemanticRecord<T>
{
    [SetsRequiredMembers]
    public SemanticRecord(T item, float[] embedding)
    {
        Item = item;
        Embedding = embedding;
    }

    public required T Item { get; set; }

    public required float[] Embedding { get; set; }
}
