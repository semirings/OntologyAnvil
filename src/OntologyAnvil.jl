"""
    OntologyAnvil

A dedicated, matrix-native rules engine for DoubleNaught's prompting ontologies.

Architecture (per hand-off):
  1. Data source boundary: OntologyAnvil never parses LinkML. Its only input is a
     D4M.Assoc already built from the (rowKey, colKey, val) triples compiled out of
     *.parquet -- today that Assoc is constructed on the Julia side of the
     d4m_juliacall_bridge, from rows/cols/vals a Python caller read out of parquet.
     Nothing in this module touches a .yaml or .parquet file directly; that boundary
     stays upstream, and stays that way even if a native Julia parquet reader is
     added later -- it would still hand this module a plain Assoc, not a file path.
  2. Matrix-native execution: every check is a sparse linear-algebra expression
     (indicator vectors, masked elementwise AND/NOT, quadratic forms) over the
     Boolean semiring -- no procedural loops or if/else trees over entities or slots.
  3. Strict type stability: the engine struct is parametric in its sparse element
     type; every function dispatches on concrete types. The one inherent exception
     is the String=>Int label maps (class_index / slot_index) -- there is no way to
     avoid a Dict here without fixing the vocabulary at compile time, which a
     runtime-loaded ontology rules out.
  4. Modularity: loading (`build_engine`), class-exclusivity checking
     (`check_class_assignment`), and slot-requirement checking (`check_requirements`)
     are independent, composable functions over one immutable engine struct, so a
     third check (e.g. any_of/range-membership validation for slots like
     `carried_by`) can be added as a fourth function without touching the others.

Semantics, corrected against the real prompt_governance_ontology triples (NOT the
flat single-vocabulary co-occurrence model of the original straw man):

  - `disjointWith|X` triples are CLASS-exclusivity facts: (Prop, disjointWith|Character,
    Character) means one entity cannot be typed as both Prop and Character at once.
    This is checked per-entity, over the universe of NarrativeEntity subclasses --
    not a flat co-occurrence check across every token in a prompt.
  - `requires|X` triples are PER-CLASS SLOT requirements: (Prop, requires|entity_name,
    entity_name) means every Prop instance must have entity_name filled. This is a
    class x slot incidence relation, not a class x class adjacency relation, so it
    gets its own matrix shaped (classes x slots), not reused from the disjoint matrix.

Not yet covered (left as a clean extension point, not silently assumed correct):
  - any_of / union-range validation (e.g. carried_by's filler must be a Character or
    a Creature) -- this needs the candidate's object-valued slot fillers to carry
    their own class label, which the extraction side of the Ontology Guardrail node
    doesn't yet produce. Wire it in as `check_range_constraint` once that exists.
"""
module OntologyAnvil

using SparseArrays
using LinearAlgebra
using D4M: Assoc, find   # the only D4M surface this module touches -- an already-
                         # built Assoc, read via its public `find` accessor. Nothing
                         # here constructs or mutates an Assoc; that stays upstream.

export OntologyAnvilEngine, build_engine, check_class_assignment, check_requirements,
       validate_entity

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

"""
    OntologyAnvilEngine{Tv<:Real}

Immutable, pre-compiled view of an ontology Assoc as sparse matrices.
`Tv` is the element type of both sparse matrices (Bool by default -- these are
pure indicator relations, not weighted ones. A future weighted-rule variant could
reuse this same struct with Tv = Float64.).

Label maps (`class_index`, `slot_index`) are the one inherent String=>Int boundary:
built once at `build_engine` time, never mutated, never touched inside a hot check.
"""
struct OntologyAnvilEngine{Tv<:Real}
    class_index::Dict{String,Int}
    slot_index::Dict{String,Int}
    classes::Vector{String}            # class_index inverse, for reporting
    slots::Vector{String}              # slot_index inverse, for reporting
    disjoint_matrix::SparseMatrixCSC{Tv,Int}      # classes x classes, symmetric
    requirement_matrix::SparseMatrixCSC{Tv,Int}   # slots x classes (TRANSPOSED from
                                                   # the natural classes x slots reading
                                                   # -- see build_engine's note below)
end

const DISJOINT_PREFIX = "disjointWith|"
const REQUIRES_PREFIX = "requires|"

"""
    build_engine(ontology::Assoc; Tv::Type{<:Real}=Bool) -> OntologyAnvilEngine{Tv}

Compile an ontology Assoc (already loaded by D4M from the parquet-derived triples)
into the sparse matrices the checks run against. `Assoc` is D4M.jl's type -- this
function is written against its public `find`/`getrow`/`getcol` accessors, not
against any parquet- or YAML-reading code.

Single pass over `find(ontology)`: classifies each triple by its Col prefix and
appends into one of two label-pair lists, then builds both sparse matrices in one
vectorized `sparse(...)` call each -- no per-triple matrix mutation, no loops with
branching logic beyond the one dispatch on prefix.
"""
function build_engine(ontology::Assoc; Tv::Type{<:Real}=Bool)
    rows, cols, _vals = find(ontology)  # D4M.Assoc accessor; row/col/val triples

    disjoint_pairs = Tuple{String,String}[]
    requires_pairs = Tuple{String,String}[]

    for (r, c) in zip(rows, cols)
        cs = String(c)
        if startswith(cs, DISJOINT_PREFIX)
            other = cs[length(DISJOINT_PREFIX)+1:end]
            push!(disjoint_pairs, (String(r), other))
        elseif startswith(cs, REQUIRES_PREFIX)
            slot = cs[length(REQUIRES_PREFIX)+1:end]
            push!(requires_pairs, (String(r), slot))
        end
        # hasAttribute|X, optional|X, is_a, rule, hasValue|X, carried_by|X etc. are
        # out of scope for this engine's two checks and are intentionally skipped --
        # not an oversight, see the module docstring's "not yet covered" note for
        # carried_by's any_of facets specifically.
    end

    classes = sort(unique(vcat(
        (p[1] for p in disjoint_pairs)...,
        (p[2] for p in disjoint_pairs)...,
        (p[1] for p in requires_pairs)...,
    )))
    slots = sort(unique(p[2] for p in requires_pairs))

    class_index = Dict(c => i for (i, c) in enumerate(classes))
    slot_index = Dict(s => i for (i, s) in enumerate(slots))
    nC = length(classes)
    nS = length(slots)

    # Disjointness: symmetric incidence over classes x classes.
    di = [class_index[a] for (a, _b) in disjoint_pairs]
    dj = [class_index[b] for (_a, b) in disjoint_pairs]
    disjoint_matrix = if isempty(di)
        spzeros(Tv, nC, nC)
    else
        D = sparse(di, dj, ones(Tv, length(di)), nC, nC, (a, b) -> a | b)
        D = D .| permutedims(D)   # enforce symmetry regardless of source ordering
        D
    end

    # Requirements: incidence between classes and slots -- but stored as
    # (slots x classes), transposed from how it reads (class requires slot). A
    # per-class lookup is the hot path (one per candidate entity check), and
    # SparseMatrixCSC is column-major: engine.requirement_matrix[:, c] is a native
    # O(nnz in that column) slice, where the natural classes-x-slots orientation
    # would force a row slice -- a column-major scan across every stored column
    # just to pull out one row. Storing it pre-transposed makes the hot path a
    # real column slice instead of a row-slice workaround.
    ri = [slot_index[s] for (_a, s) in requires_pairs]
    rj = [class_index[a] for (a, _s) in requires_pairs]
    requirement_matrix = if isempty(ri)
        spzeros(Tv, nS, nC)
    else
        sparse(ri, rj, ones(Tv, length(ri)), nS, nC, (a, b) -> a | b)
    end

    return OntologyAnvilEngine{Tv}(
        class_index, slot_index, classes, slots, disjoint_matrix, requirement_matrix
    )
end

# ---------------------------------------------------------------------------
# Indicator-vector helper
# ---------------------------------------------------------------------------

"""
Build a sparse Bool/Tv indicator vector over `index` from `labels`, silently
ignoring any label not in the engine's vocabulary (an unrecognized class or slot
name is a data-quality finding for the extraction step upstream, not this
engine's concern -- it has nothing to check an unknown label against)."""
function _indicator(index::Dict{String,Int}, labels, ::Type{Tv}) where {Tv<:Real}
    n = length(index)
    idx = [index[l] for l in labels if haskey(index, l)]
    isempty(idx) && return spzeros(Tv, n)
    return sparsevec(idx, ones(Tv, length(idx)), n)
end

# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

"""
    check_class_assignment(engine, asserted_classes) -> (is_valid::Bool, violations::Vector{String})

Validates that `asserted_classes` (the class label(s) a candidate entity has been
tagged with by the extraction step) does not contain two mutually-disjoint classes
at once. One sparse quadratic form plus one masked elementwise product -- no
pairwise loop over candidate classes.
"""
function check_class_assignment(engine::OntologyAnvilEngine{Tv}, asserted_classes) where {Tv}
    m = _indicator(engine.class_index, asserted_classes, Tv)

    score = dot(m, engine.disjoint_matrix * m)
    if iszero(score)
        return true, String[]
    end

    # Which of the asserted classes are the offending ones? A class c is offending
    # iff it's asserted (m[c] != 0) AND it conflicts with something else asserted
    # (disjoint_matrix * m)[c] != 0 -- one elementwise product, then read off the
    # nonzero indices.
    offending = (engine.disjoint_matrix * m) .* m
    bad_idx = findall(!iszero, offending)
    bad_classes = engine.classes[bad_idx]
    return false, ["Disjointness violation: entity asserts mutually exclusive classes among $(bad_classes)"]
end

"""
    check_requirements(engine, class_label, asserted_slots) -> (is_valid::Bool, violations::Vector{String})

Validates that every slot `class_label` requires (per `requires|X` facts) is
present in `asserted_slots` (the slots the extraction step actually filled in for
this candidate entity). One sparse row slice, one elementwise AND-NOT, no loop
over the slot universe.
"""
function check_requirements(engine::OntologyAnvilEngine{Tv}, class_label::AbstractString, asserted_slots) where {Tv}
    if !haskey(engine.class_index, class_label)
        return false, ["Unknown class: $(class_label) is not in the loaded ontology"]
    end
    c = engine.class_index[class_label]
    f = _indicator(engine.slot_index, asserted_slots, Tv)

    required_col = engine.requirement_matrix[:, c]  # native CSC column slice -- see
                                                     # the storage-orientation note
                                                     # on OntologyAnvilEngine's field.
    missing_vec = required_col .& .!Bool.(f)   # AND-NOT over the Boolean semiring
    missing_idx = findall(!iszero, missing_vec)
    if isempty(missing_idx)
        return true, String[]
    end
    missing_slots = engine.slots[missing_idx]
    return false, ["Missing required slot(s) for $(class_label): $(missing_slots)"]
end

"""
    validate_entity(engine, asserted_classes, asserted_slots) -> (is_valid::Bool, violations::Vector{String})

The actual call the Ontology Guardrail node makes: one candidate entity's asserted
class label(s) plus the slots the extraction step filled in, validated in one shot.

Composes `check_class_assignment` and `check_requirements` rather than duplicating
either's logic, but the composition itself encodes a real rule, not just plumbing:
slot requirements are only meaningful once the entity resolves to exactly ONE
recognized class. Zero classes (extraction found nothing to type it as) or more
than one (which `check_class_assignment` may itself flag as a disjointness
violation, or may not -- two asserted classes that happen not to be disjoint, e.g.
a bug upstream asserting both "Prop" and some future non-disjoint class) both leave
no single class to check slot requirements against, and that absence of a
checkable class is itself reported rather than silently skipped.
"""
function validate_entity(engine::OntologyAnvilEngine, asserted_classes, asserted_slots)
    class_ok, class_violations = check_class_assignment(engine, asserted_classes)
    violations = copy(class_violations)

    recognized = unique(c for c in asserted_classes if haskey(engine.class_index, c))
    if length(recognized) != 1
        push!(
            violations,
            "Cannot check slot requirements: entity must resolve to exactly one " *
            "recognized class, got $(recognized) from asserted classes $(asserted_classes)",
        )
        return false, violations
    end

    _req_ok, req_violations = check_requirements(engine, recognized[1], asserted_slots)
    append!(violations, req_violations)
    return isempty(violations), violations
end

end # module OntologyAnvil
