using System.Diagnostics.CodeAnalysis;

namespace ChatAIze.SemanticIndex;

/// <summary>
/// Represents an item stored in the <see cref="SemanticDatabase{T}"/> along
/// with its vector embedding.
/// </summary>
/// <typeparam name="T">The type of the stored item.</typeparam>
public record SemanticRecord<T>
{
    /// <summary>
    /// Initializes a new instance of the <see cref="SemanticRecord{T}"/> class
    /// with the specified item and embedding.
    /// </summary>
    /// <param name="item">The item to store.</param>
    /// <param name="embedding">The vector embedding associated with the
    /// item.</param>
    [SetsRequiredMembers]
    public SemanticRecord(T item, float[] embedding)
    {
        Item = item;
        Embedding = embedding;
    }

    /// <summary>
    /// Gets or sets the stored item.
    /// </summary>
    public virtual required T Item { get; set; }

    /// <summary>
    /// Gets or sets the vector embedding for the item.
    /// </summary>
    public virtual required float[] Embedding { get; set; }
}
