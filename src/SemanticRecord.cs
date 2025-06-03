using System.Diagnostics.CodeAnalysis;

namespace ChatAIze.SemanticIndex;

public record SemanticRecord<T>
{
    [SetsRequiredMembers]
    public SemanticRecord(T item, float[] embedding)
    {
        Item = item;
        Embedding = embedding;
    }

    public virtual required T Item { get; set; }

    public virtual required float[] Embedding { get; set; }
}
