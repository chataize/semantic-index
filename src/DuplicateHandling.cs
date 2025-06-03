namespace ChatAIze.SemanticIndex;

/// <summary>
/// Defines how the <see cref="SemanticDatabase{T}"/> should handle duplicate
/// items when adding records.
/// </summary>
public enum DuplicateHandling
{
    /// <summary>
    /// Always add the item even if a duplicate already exists.
    /// </summary>
    Allow,

    /// <summary>
    /// Replace the existing record when a duplicate is added.
    /// </summary>
    Update,

    /// <summary>
    /// Ignore the new item when a duplicate exists.
    /// </summary>
    Skip,

    /// <summary>
    /// Throw an <see cref="InvalidOperationException"/> if a duplicate exists.
    /// </summary>
    Throw
}
