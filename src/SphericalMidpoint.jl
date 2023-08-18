using DifferentialEquations
using Plots

#Constants and setup


mutable struct SphericalMP
    si::Matrix{Float64}
    sf1::Matrix{Float64}
    sf2::Matrix{Float64}
    sm::Matrix{Float64}
    lattice::Lattice
    Hsm::Vector{Vector{Float64}}
    structureFactors::Matrix{Vector{ComplexF64}}

    phononMomentaPrev::Matrix{Float64}
    phononMomenta::Matrix{Float64}
    phononMass::Vector{Float64}
    phononDamp::Vector{Float64}
    phononDrive::Vector{Function}
    dt::Float64

    obs::EvolveObservables

    SphericalMP()=new{}()
end


function initSphereMP(dim,lattice,phdim)
    smp=SphericalMP()
    smp.lattice=deepcopy(lattice)
    smp.si = deepcopy(smp.lattice.expVals)
    smp.sf1 = Array{Float64,2}(undef, dim^2, lattice.length)
    smp.sf2 = Array{Float64,2}(undef, dim^2, lattice.length)
    smp.sm = Array{Float64,2}(undef, dim^2, lattice.length)
    smp.Hsm = Vector{Vector{Float64}}(undef, lattice.length) 
    smp.structureFactors = Matrix{Vector{ComplexF64}}(undef,dim^2,dim^2)

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

function setsi!(smp,site,newState)
    smp.si[:,site] = newState
end

function getsi(smp,site)
    return(smp.si[:,site])
end

function setsf1!(smp,site,newState)
    smp.sf1[:,site] = newState
end

function getsf1(smp,site)
    return(smp.sf1[:,site])
end

function setsf2!(smp,site,newState)
    smp.sf1[:,site] = newState
end

function getsf2(smp,site)
    return(smp.sf1[:,site])
end

function setsm!(smp,site,newState)
    smp.sm[:,site] = newState
end

function getsm(smp,site)
    return(smp.sm[:,site])
end

function setDT!(smp,dt)
    smp.dt=dt
end

function setStructureFactors!(smp,gens,dim)
    mats=copy(gens.generators)

    for i in 1:dim^2
        for j in 1:dim^2
            res=mats[i]*mats[j]-mats[j]*mats[i]
            vec=decomposeMat(gens,res)
            smp.structureFactors[i,j]=-1im*vec

        end
    end
end

function updateHsm!(smp, lattice, gens)
    for site in 1:length(lattice)
        interactionSites = getInteractionSites(lattice, site)
        interactionMatrices = getInteractionMatrices(lattice, site)
        output=zeros(gens.dim^2)
        mat=zeros(gens.dim^2,gens.dim^2)
        s0 = getsm(smp, site)
        for i in 1:gens.dim^2
            for j in 1:gens.dim^2
                mat[i,j]=dot(smp.structureFactors[i,j], s0)
            end
        end
        for k in 1:length(interactionSites)
            Jex=interactionMatrices[k].mat
            s1 = getsm(smp, interactionSites[k])
            output += Jex*mat*s1
        end
        smp.Hsm[site]=output
    end
end

function evolveSpinSphereMP!(smp,lattice,gens)

    for site in 1:length(lattice)
        s0 = getsi(smp,site)
        sf = getsf1(smp,site)

        sm = (sf[1:3] + s0[1:3])./norm(sf[1:3] + s0[1:3])
        push!(sm, 1.0)
        setsm!(smp,site,sm)
    end

    updateHsm!(smp,lattice,gens)

    for site in 1:length(lattice)
        s0 = getsi(smp,site)
        sm = getsm(smp,site)

        sf = s0 + (smp.dt*smp.Hsm[site])
        setsf1!(smp,site,sf)
    end

end

function evolveSphereMP!(smp,lattice,gens)
    # set Zf1 to Z0 for first iteration
    for site in 1:length(lattice)
        s0 = getsi(smp,site)
        setsf1!(smp,site,s0)
    end

    iterations=10
    for i in 1:iterations
        evolveSpinSphereMP!(smp,lattice,gens)
    end

    for site in 1:length(lattice)
        s = getsf1(smp,site)
        setsi!(smp,site,s)
        smp.lattice.expVals[:,site] = s
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
