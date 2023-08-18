using DifferentialEquations
using Plots
#Constants and setup
mutable struct SchrodingerMP
    Zi::Matrix{ComplexF64}
    Zf1::Matrix{ComplexF64}
    Zf2::Matrix{ComplexF64}
    Zm::Matrix{ComplexF64}
    lattice::Lattice
    HZm::Vector{Matrix{ComplexF64}}
    phononMomentaPrev::Matrix{Float64}
    phononMomenta::Matrix{Float64}
    phononMass::Vector{Float64}
    phononDamp::Vector{Float64}
    phononDrive::Vector{Function}
    dt::Float64
    obs::EvolveObservables
    SchrodingerMP()=new{}()
end
function initSMP(dim,lattice,phdim)
    smp=SchrodingerMP()
    smp.lattice=deepcopy(lattice)
    smp.Zi = deepcopy(smp.lattice.spins)
    smp.Zf1 = Array{Float64,2}(undef, dim, lattice.length)
    smp.Zf2 = Array{Float64,2}(undef, dim, lattice.length)
    smp.Zm = Array{Float64,2}(undef, dim, lattice.length)
    smp.HZm = Vector{Matrix{ComplexF64}}(undef, lattice.length) 
    smp.phononMass = zeros(phdim)
    smp.phononDamp = zeros(phdim)
    smp.phononDrive = Vector{Function}(undef,phdim)
    smp.phononMomenta = Array{Float64,2}(undef, phdim, lattice.length)
    smp.phononMomentaPrev = Array{Float64,2}(undef, phdim, lattice.length)
    smp.dt = 0.05
    smp.obs=initEvolveObservables()
    #Function that returns 0 for all times if no drive is specified
    function noDrive(t)
        x=0
        return(x)
    end
    for i in 1:phdim
        smp.phononDrive[i]=noDrive
    end
    
    return smp
end
function setZi!(smp,site,newState)
    smp.Zi[:,site] = newState
end
function getZi(smp,site)
    return(smp.Zi[:,site])
end
function setZf1!(smp,site,newState)
    smp.Zf1[:,site] = newState
end
function getZf1(smp,site)
    return(smp.Zf1[:,site])
end
function setZf2!(smp,site,newState)
    smp.Zf1[:,site] = newState
end
function getZf2(smp,site)
    return(smp.Zf1[:,site])
end
function setZm!(smp,site,newState)
    smp.Zm[:,site] = newState
end
function getZm(smp,site)
    return(smp.Zm[:,site])
end
function setDT!(smp,dt)
    smp.dt=dt
end
function updateHZm!(smp, lattice, gens)
    for site in 1:length(lattice)
        interactionSites = getInteractionSites(lattice, site)
        interactionMatrices = getInteractionMatrices(lattice, site)
        scale=zeros(ComplexF64,gens.dim^2)
        for i in 1:length(interactionSites)
            s1 = genExpVals(getZm(smp, interactionSites[i]), gens)
            Jex=interactionMatrices[i].mat
            scale+=Jex*s1
        end
        smp.HZm[site]=sum(scale.*gens.generators)
    end
end
function evolveSpinSMP!(smp,lattice,gens)
    for site in 1:length(lattice)
        Z0 = getZi(smp,site)
        Zf = getZf1(smp,site)
        Zm = 0.5*(Zf + Z0)
        setZm!(smp,site,Zm)
    end
    updateHZm!(smp,lattice,gens)
    for site in 1:length(lattice)
        Z0 = getZi(smp,site)
        Zm = getZm(smp,site)
        Zf = Z0 - ((1im*smp.dt)*(smp.HZm[site]*Zm))
        setZf1!(smp,site,Zf)
    end
end
function evolveSMP!(smp,lattice,gens)
    # set Zf1 to Z0 for first iteration
    for site in 1:length(lattice)
        Z0 = getZi(smp,site)
        setZf1!(smp,site,Z0)
    end
    iterations=10
    for i in 1:iterations
        evolveSpinSMP!(smp,lattice,gens)
    end
    for site in 1:length(lattice)
        Z = getZf1(smp,site)
        setZi!(smp,site,Z)
        setSpin!(lattice, site, Z)
    end
end
# initi
function initPhMomentum!(evs,T,phd)
    for site in 1:length(evs.lattice)
        P=rand(Uniform(0,1),phd)
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