using DifferentialEquations
using Plots

#Constants and setup


mutable struct Evolution
    structureFactors::Matrix{Vector{ComplexF64}}
    latticePrev::Lattice
    lattice::Lattice
    phononMomentaPrev::Matrix{Float64}
    phononMomenta::Matrix{Float64}
    phononMass::Vector{Float64}
    phononDamp::Vector{Float64}
    tspan::Tuple

    Evolution()=new{}()
end


function initEv(dim,lattice,gens,timeStep,phdim)
    evs = Evolution()
    finalState!(lattice,gens,dim)
    evs.structureFactors = Matrix{Vector{ComplexF64}}(undef,dim^2,dim^2)
    evs.lattice = lattice
    evs.latticePrev = lattice
    evs.tspan = timeStep
    evs.phononMass = zeros(phdim)
    evs.phononDamp = zeros(phdim)
    evs.phononMomenta = Array{Float64,2}(undef, phdim, lattice.length)
    evs.phononMomentaPrev = Array{Float64,2}(undef, phdim, lattice.length)


    return evs
end

function initPhMomentum!(evs,T,phd)
    for site in 1:length(evs.lattice)
        P=rand(Uniform(0,1),2)
        p0=getPhonon(evs.latticePrev,site)


        A=-2*T*log.(P)
        B=dot(evs.lattice.springConstants,p0.^2)
        v=sqrt.((A .+ B).*evs.phononMass)

        choice=[1.0,-1.0]
        sign=rand(choice,phd)

        v.*=sign


        setPhononMomentum!(evs,site,v)
    end


end

function getExpValSpin(evs, site)
    return evs.latticePrev.expVals[:,site]
end

function setExpValSpin!(evs, site, newState)
    evs.lattice.expVals[:,site] = newState
end

function setPhononMass!(evs,vec,phd)
    length(vec) == (phd) || error(string("Phonon mass must be of size ",phd,"."))
    evs.phononMass = vec
end

function setPhononDamp!(evs,vec,phd)
    length(vec) == (phd) || error(string("Phonon damping constants must be of size ",phd,"."))
    evs.phononDamp = vec
end

function setPhononMomentum!(evs, site::Int, newState::Vector{Float64}) 
    evs.phononMomenta[:,site] = newState
end

function setStructureFactors!(evs,gens,dim)
    mats=copy(gens.generators)
    Id=Matrix((1.0+0im)I,dim,dim)
    push!(mats,Id)

    for i in 1:dim^2
        for j in 1:dim^2
            res=mats[i]*mats[j]-mats[j]*mats[i]
            vec=decomposeMat(gens,res,dim)
            evs.structureFactors[i,j]=1im*vec

        end
    end
end


function evolve_spin(gens,evs,site,dim)
    Id=Matrix((1.0+0im)I,dim,dim)

    s0=getExpValSpin(evs,site)
    p0 = getPhonon(evs.latticePrev, site)
    

    interactionSites = getInteractionSites(evs.latticePrev, site)
    interactionMatrices = getInteractionMatrices(evs.latticePrev, site)
    interactionField = getInteractionField(evs.latticePrev, site)
    output=zeros(dim^2)
    for j in 1:length(interactionSites)
        Jex=interactionMatrices[j].mat
        s1 = getExpValSpin(evs,interactionSites[j])

        output+=Jex*s1
    end


    function update(xdot,x,p,t)
        mat=zeros(dim^2,dim^2)
        for i in 1:dim^2
            for j in 1:dim^2
                mat[i,j]=dot(evs.structureFactors[i,j], x)
            end
        end

        coupling = evs.lattice.phononCoupling
        vec = coupling*p0

        xdot[:] = mat*output + mat*vec + mat*interactionField
    end


    spin_prob = ODEProblem(update,s0,evs.tspan)
    sol = solve(spin_prob)



    return (last(sol.u))

end

function evolve_phonon(gens,evs,site,dim,phdim)
    x0=getPhonon(evs.latticePrev,site)
    p0=getPhononMomentum(evs,site)

    append!(x0,p0)

    s0= getExpValSpin(evs,site)

    coupling = evs.lattice.phononCoupling
    springConst = evs.lattice.springConstants



    vec=transpose(coupling)*s0

    

    function update(xdot,x,p,t)
        xdot[1:phdim] = x[phdim+1:end]./(evs.phononMass)
        xdot[phdim+1:end] = -springConst.*x[1:phdim]-vec
    end


    phononProb = ODEProblem(update,x0,evs.tspan)
    sol = solve(phononProb)

    return (last(sol.u))
end



function evolve!(evs,gens,dim,phdim,T,numSteps)

    for i in 1:numSteps
        for site in 1:length(evs.lattice)
            spin=evolve_spin(gens,evs,site,dim)
            phonon=evolve_phonon(gens,evs,site,dim,phdim)
            setExpValSpin!(evs,site,spin)
            setPhonon!(evs.lattice,site,phonon[1:phdim])
            setPhononMomentum!(evs,site,phonon[phdim+1:end])
        end
        evs.phononMomentaPrev = copy(evs.phononMomenta)
        evs.latticePrev = deepcopy(evs.lattice)

    end





end