
mutable struct HeunP
    Z1::Matrix{ComplexF64}
    Z2::Matrix{ComplexF64}
    lattice::Lattice
    HZ::Vector{Matrix{ComplexF64}}
    HZ1::Vector{Matrix{ComplexF64}}

    phononMomentaPrev::Matrix{Float64}
    phononMomenta::Matrix{Float64}
    phononMass::Vector{Float64}
    phononDamp::Vector{Float64}
    phononDrive::Vector{Function}
    dt::Float64
    obs::EvolveObservables
    
    HeunP()=new{}()
end

function initHeunP(dim,lattice,phdim)
    hp=HeunP()
    hp.lattice=deepcopy(lattice)
    hp.Z1 = Array{Float64,2}(undef, dim, lattice.length)
    hp.Z2 = Array{Float64,2}(undef, dim, lattice.length)
    hp.HZ = Vector{Matrix{ComplexF64}}(undef, lattice.length) 
    hp.HZ1 = Vector{Matrix{ComplexF64}}(undef, lattice.length) 


    hp.phononMass = zeros(phdim)
    hp.phononDamp = zeros(phdim)
    hp.phononDrive = Vector{Function}(undef,phdim)
    hp.phononMomenta = Array{Float64,2}(undef, phdim, lattice.length)
    hp.phononMomentaPrev = Array{Float64,2}(undef, phdim, lattice.length)
    hp.dt = 0.05
    hp.obs=initEvolveObservables()

    #Function that returns 0 for all times if no drive is specified
    function noDrive(t)
        x=0
        return(x)
    end

    for i in 1:phdim
        hp.phononDrive[i]=noDrive
    end
    



    return (hp)
end

function setPhononMass!(hp,vec,phd)
    length(vec) == (phd) || error(string("Phonon mass must be of size ",phd,"."))
    hp.phononMass = vec
end

function setPhononDamp!(hp,vec,phd)
    length(vec) == (phd) || error(string("Phonon damping constants must be of size ",phd,"."))
    hp.phononDamp = vec
end

function setPhononMomentum!(hp, site::Int, newState::Vector{Float64}) 
    hp.phononMomenta[:,site] = newState
end


function initPhMomentum!(hp,T,phd)
    for site in 1:length(hp.lattice)
        P=rand(Uniform(0,1),phd)
        p0=getPhonon(hp.latticePrev,site)


        A=-2*T*log.(P)
        B=dot(hp.lattice.springConstants,p0.^2)
        v=sqrt.((A .+ B).*hp.phononMass)

        choice=[1.0,-1.0]
        sign=rand(choice,phd)

        v.*=sign


        setPhononMomentum!(hp,site,v)
    end


end




function setZ1!(hp,site,newState)
    hp.Z1[:,site] = newState
end

function getZ1!(hp,site)
    return(hp.Z1[:,site])
end

function setZ2!(hp,site,newState)
    hp.Z2[:,site] = newState
end

function getZ2!(hp,site)
    return(hp.Z2[:,site])
end

function setDT!(hp,dt)
    hp.dt=dt
end

function updateHZ!(hp, gens)

    for site in 1:length(hp.lattice)

        interactionSites = getInteractionSites(hp.lattice, site)
        interactionMatrices = getInteractionMatrices(hp.lattice, site)
        scale=zeros(ComplexF64,gens.dim^2)
        for i in 1:length(interactionSites)
            s1 = genExpVals(getSpin(hp.lattice, interactionSites[i]), gens)
            Jex=interactionMatrices[i].mat
            scale+=Jex*s1
        end
        hp.HZ[site]=sum(scale.*gens.generators)
    end
end

function updateHZ1!(hp, gens)

    for site in 1:length(hp.lattice)

        interactionSites = getInteractionSites(hp.lattice, site)
        interactionMatrices = getInteractionMatrices(hp.lattice, site)
        scale=zeros(ComplexF64,gens.dim^2)
        for i in 1:length(interactionSites)
            s1 = genExpVals(getZ1!(hp, interactionSites[i]), gens)
            Jex=interactionMatrices[i].mat
            scale+=Jex*s1
        end
        hp.HZ1[site]=sum(scale.*gens.generators)
    end
end

function evolveSpinHP!(hp,gens)
    updateHZ!(hp, gens)
    for site in 1:length(hp.lattice)
        Z0=getSpin(hp.lattice,site)
        newState=Z0-1im*hp.dt*hp.HZ[site]*Z0
        setZ1!(hp,site,newState)
    end

    updateHZ1!(hp,gens)
    for site in 1:length(hp.lattice)
        Z0=getSpin(hp.lattice,site)
        Z1=getZ1!(hp,site)
        newState=Z0-((1im*hp.dt/2)*(hp.HZ[site]*Z0+hp.HZ1[site]*Z1))
        setZ2!(hp,site,newState)
    end
end


function evolveHP!(hp,gens)
    evolveSpinHP!(hp,gens)
    for site in 1:length(hp.lattice)
        Z2=getZ2!(hp,site)
        Z2/=norm(Z2)
        setSpin!(hp.lattice,site,Z2)
    end
end
    









