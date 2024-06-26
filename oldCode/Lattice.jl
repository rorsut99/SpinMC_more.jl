
# Added dim variable to define dimension of spin vector

mutable struct Lattice{D,N,dim,phdim}
    size::NTuple{D, Int} #linear extent of the lattice in number of unit cells
    length::Int #Number of sites N_sites
    unitcell::UnitCell{D}
    sitePositions::Vector{NTuple{D,Float64}}

    spins::Matrix{ComplexF64} #3*N_sites matrix containing the spin configuration
    phonons::Matrix{Float64}
    phononCoupling::Matrix{Float64}
    springConstants::Vector{Float64}
    Qmax::Float64

    interactionSites::Vector{NTuple{N,Int}} #list of length N_sites, for every site contains all interacting sites
    interactionMatrices::Vector{NTuple{N,InteractionMatrix}} #list of length N_sites, for every site contains all interaction matrices
    interactionOnsite::Vector{InteractionMatrix} #list of length N_sites, for every site contains the local onsite interaction matrix
    interactionField::Vector{Any} #list of length N_sites, for every site contains the local field
    generators::Vector{Matrix{ComplexF64}} #list of length dim^2-1 holding generators for the SU(N) representation of interest

    spinOperators::Vector{Matrix{ComplexF64}}
    genReps::Matrix{Vector{ComplexF64}}

    expVals::Vector{Vector{Float64}}
    Lattice(D,N,dim,phdim) = new{D,N,dim,phdim}()
end


# Added dim variable to define dimension of spin vector
function Lattice(uc::UnitCell{D}, L::NTuple{D,Int},dim::Int, phdim::Int) where D
    #parse interactions
    ##For every basis site b, generate list of sites which b interacts with and store the corresponding interaction sites and matrices. 
    ##Interaction sites are specified by the target site's basis id, b_target, and the offset in units of primitive lattice vectors. 
    ##If b has multiple interactions defined with the same target site, eliminate those duplicates by summing up the interaction matrices. 
    interactionTargetSites = [ Vector{Tuple{Int,NTuple{D,Int},Matrix{Float64}}}(undef,0) for i in 1:length(uc.basis) ] #tuples of (b_target, offset, M)
    for x in uc.interactions
        b1, b2, offset, M = x
        b1 == b2 && offset == Tuple(zeros(D)) && error("Interaction cannot be local. Use setInteractionOnsite!() instead.")
        
        #locate existing coupling to target site and add interaction matrix
        for i in 1:length(interactionTargetSites[b1])
            if interactionTargetSites[b1][i][1] == b2 && interactionTargetSites[b1][i][2] == offset
                interactionTargetSites[b1][i] = (interactionTargetSites[b1][i][1], interactionTargetSites[b1][i][2], interactionTargetSites[b1][i][3] + M)
                @goto endb1
            end
        end
        #if coupling does not exist yet, push new entry
        push!(interactionTargetSites[b1], (b2, offset, M))
        @label endb1

        #locate existing coupling from target site and add interaction matrix
        for i in 1:length(interactionTargetSites[b2])
            if interactionTargetSites[b2][i][1] == b1 && interactionTargetSites[b2][i][2] == (x->-x).(offset)
                interactionTargetSites[b2][i] = (interactionTargetSites[b2][i][1], interactionTargetSites[b2][i][2], interactionTargetSites[b2][i][3] + transpose(M))
                @goto endb2
            end
        end
        #if coupling does not exist yet, push new entry
        push!(interactionTargetSites[b2], (b1, (x->-x).(offset), transpose(M)))
        @label endb2
    end
    Ninteractions = findmax([ length(interactionTargetSites[i]) for i in 1:length(uc.basis) ])[1]

    #create lattice struct
    lattice = Lattice(D,Ninteractions,dim,phdim)
    lattice.size = L
    lattice.length = prod(L) * length(uc.basis)
    lattice.unitcell = uc

    #generate linear representation of lattice sites to assign integer site IDs
    ##Enumeration sequence is (a1, a2, ..., b) in row-major fashion
    sites = Vector{NTuple{D+1,Int}}(undef, lattice.length)
    function nextSite(site)
        next = collect(site)
        next[D+1] += 1
        if next[D+1] > length(uc.basis)
            next[D+1] = 1
            next[D] += 1
        end
        for d in reverse(1:D)
            if next[d] >= L[d]
                next[d] = 0
                d-1 > 0 && (next[d-1] += 1)
            end
        end
        return Tuple(next)
    end
    sites[1] = tuple(zeros(Int,D)..., 1)
    for i in 2:length(sites)
        sites[i] = nextSite(sites[i-1])
    end

    #init site positions
    lattice.sitePositions = Vector{NTuple{D,Float64}}(undef, length(sites))
    for i in 1:length(sites)
        site = sites[i]
        lattice.sitePositions[i] = .+([uc.primitive[j] .* site[j] for j in 1:D]...) .+ uc.basis[site[end]]
    end

    #init spins 
    #Updated dimension of spins object
    lattice.spins = Array{Float64,2}(undef, dim, length(sites))
    lattice.phonons = Array{Float64,2}(undef, phdim, length(sites))

    lattice.springConstants = zeros(phdim)
    lattice.genReps =  Matrix{Vector{ComplexF64}}(undef,4,3)

    lattice.phononCoupling =  Array{Float64,2}(undef, phdim, dim^2-1)

    #write interactions to lattice
    lattice.interactionSites = repeat([ NTuple{Ninteractions,Int}(ones(Int,Ninteractions)) ], lattice.length)
    lattice.interactionMatrices = repeat([ NTuple{Ninteractions,InteractionMatrix}(repeat([InteractionMatrix(dim)],Ninteractions)) ], lattice.length)
    lattice.interactionOnsite = repeat([InteractionMatrix(dim)], lattice.length)
    lattice.interactionField = repeat( [Tuple(zeros(dim^2-1))], lattice.length)

    function applyPBC(n, L)
        while n < 0; n += L end
        while n >= L; n -= L end
        return n
    end
    function siteIndexFromParametrization(site)
       return findfirst(isequal(site), sites) 
    end

    for i in 1:length(sites)
        site = sites[i]
        b = site[end]

        #onsite interaction
        lattice.interactionOnsite[i] = InteractionMatrix(uc.interactionsOnsite[b])

        #field interaction
        lattice.interactionField[i] = NTuple{dim,Float64}(uc.interactionsField[b])

        #two-spin interactions
        interactionSites = repeat([i], Ninteractions)
        interactionMatrices = repeat([InteractionMatrix(dim)], Ninteractions)
        for j in 1:Ninteractions
            if j <= length(interactionTargetSites[b])
                b2, offset, M = interactionTargetSites[b][j]

                primitiveTarget = [applyPBC(site[k] + offset[k], L[k]) for k in 1:D]
                targetSite = tuple(primitiveTarget..., b2)

                interactionSites[j] = siteIndexFromParametrization(targetSite)
                interactionMatrices[j] = InteractionMatrix(M)
            end
        end
        lattice.interactionSites[i] = NTuple{Ninteractions,Int}(interactionSites)
        lattice.interactionMatrices[i] = NTuple{Ninteractions,InteractionMatrix}(interactionMatrices)
    end
    lattice.generators=[Matrix(1.0I,dim,dim)]
    lattice.spinOperators=[Matrix(1.0I,dim,dim)]

    #return lattice
    return lattice
end

function addGenerator!(lattice::Lattice{D,N,dim,phdim},M::Matrix{ComplexF64},d::Int64) where {D,N,dim,phdim}
    size(M) == (d,d) || error(string("Generator must be of size ",d,"x",d,"."))

    if (length(lattice.generators)==d^2-1)
        lattice.generators[1]=M

    else
        push!(lattice.generators,M)

    end
end

function addSpinOperator!(lattice::Lattice{D,N,dim,phdim},M::Matrix{ComplexF64},d::Int64) where {D,N,dim,phdim}
    if (length(lattice.spinOperators)==3)
        lattice.spinOperators[1]=M

    else
        push!(lattice.spinOperators,M)

    end
end


function addPhononInteraction!(lattice::Lattice{D,N,dim,phdim},M::Matrix{Float64},d::Int64,phd::Int64) where {D,N,dim,phdim}
    size(M) == (d^2-1,phd) || error(string("Coupling matrix must be of size ",d^2-1,"x",phd,"."))

    lattice.phononCoupling=M
end

function addSpringConstant!(lattice::Lattice{D,N,dim,phdim},vec::Vector{Float64},phd::Int64) where {D,N,dim,phdim}
    length(vec) == (phd) || error(string("Spring constants must be of size ",phd,"."))

    lattice.springConstants=vec
end


function Base.size(lattice::Lattice{D,N,dim,phdim}) where {D,N,dim,phdim}
    return lattice.size
end

function Base.length(lattice::Lattice{D,N,dim,phdim}) where {D,N,dim,phdim}
    return lattice.length
end

function getSpin(lattice::Lattice{D,N,dim,phdim}, site::Int) where {D,N,dim,phdim}
    return (lattice.spins[:,site])
end

function setSpin!(lattice::Lattice{D,N,dim,phdim}, site::Int, newState::Vector{ComplexF64}) where {D,N,dim,phdim}
    lattice.spins[:,site] = newState
end

function getSitePosition(lattice::Lattice{D,N,dim,phdim}, site::Int)::NTuple{D,Float64} where {D,N,dim,phdim}
    return lattice.sitePositions[site]
end

function getInteractionSites(lattice::Lattice{D,N,dim,phdim}, site::Int)::NTuple{N,Int} where {D,N,dim,phdim}
    return lattice.interactionSites[site]
end

function getInteractionMatrices(lattice::Lattice{D,N,dim,phdim}, site::Int)::NTuple{N,InteractionMatrix} where {D,N,dim,phdim}
    return lattice.interactionMatrices[site]
end

function getInteractionOnsite(lattice::Lattice{D,N,dim,phdim}, site::Int)::InteractionMatrix where {D,N,dim,phdim}
    return lattice.interactionOnsite[site]
end

function getInteractionField(lattice::Lattice{D,N,dim,phdim}, site::Int)::NTuple{dim,Float64} where {D,N,dim,phdim}
    return lattice.interactionField[site]
end

function setGenReps!(lattice::Lattice{D,N,dim,phdim},d) where {D,N,dim,phdim}
    for i in 1:3
        for j in 1:3
            mat=lattice.spinOperators[i]*lattice.spinOperators[j]
            vec=decomposeMat(lattice,mat,d)
            lattice.genReps[i,j]=vec
        end
    end

    for k in 1:3
        mat=lattice.spinOperators[k]
        vec=decomposeMat(lattice,mat,d)
        lattice.genReps[4,k]=vec
    end
end

