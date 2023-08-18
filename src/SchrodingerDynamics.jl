
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
        evs.phononDrive[i]=noDrive
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
        v=sqrt.((A .+ B).*evs.phononMass)

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

function updateHZ!(hp, lattice, gens)

    for site in 1:length(lattice)

        interactionSites = getInteractionSites(lattice, site)
        interactionMatrices = getInteractionMatrices(lattice, site)
        scale=zeros(ComplexF64,gens.dim^2)
        for i in 1:length(interactionSites)
            s1 = genExpVals(getSpin(lattice, interactionSites[i]), gens)
            Jex=interactionMatrices[i].mat
            scale+=Jex*s1
        end
        hp.HZ[site]=sum(scale.*gens.generators)
    end
end

function updateHZ1!(hp, lattice, gens)

    for site in 1:length(lattice)

        interactionSites = getInteractionSites(lattice, site)
        interactionMatrices = getInteractionMatrices(lattice, site)
        scale=zeros(ComplexF64,gens.dim^2)
        for i in 1:length(interactionSites)
            s1 = genExpVals(getZ1!(hp, interactionSites[i]), gens)
            Jex=interactionMatrices[i].mat
            scale+=Jex*s1
        end
        hp.HZ1[site]=sum(scale.*gens.generators)
    end
end

function evolveSpinHP!(hp,lattice,gens)
    updateHZ!(hp,lattice,gens)
    for site in 1:length(lattice)
        Z0=getSpin(lattice,site)
        newState=Z0-1im*hp.dt*hp.HZ[site]*Z0
        setZ1!(hp,site,newState)
    end

    updateHZ1!(hp,lattice,gens)
    for site in 1:length(lattice)
        Z0=getSpin(lattice,site)
        Z1=getZ1!(hp,site)
        newState=Z0-((1im*hp.dt/2)*(hp.HZ[site]*Z0+hp.HZ1[site]*Z1))
        setZ2!(hp,site,newState)
    end
end


function evolveHP!(hp,lattice,gens)
    evolveSpin!(hp,lattice,gens)
    for site in 1:length(lattice)
        Z2=getZ2!(hp,site)
        Z2/=norm(Z2)
        setSpin!(lattice,site,Z2)
    end
end
    











function initEv(dim,lattice,gens,timeStep,phdim)
    evs = Evolution()
    finalState!(lattice,gens) # save the final MC state to lattice.expVals
    evs.structureFactors = Matrix{Vector{ComplexF64}}(undef,dim^2,dim^2)
    evs.lattice = deepcopy(lattice)
    evs.latticePrev = deepcopy(lattice)
    evs.tspan = timeStep
    evs.phononMass = zeros(phdim)
    evs.phononDamp = zeros(phdim)
    evs.phononDrive = Vector{Function}(undef,phdim)
    evs.phononMomenta = Array{Float64,2}(undef, phdim, lattice.length)
    evs.phononMomentaPrev = Array{Float64,2}(undef, phdim, lattice.length)
    setTimeStep!(evs)
    evs.obs=initEvolveObservables()

    # function that is used in the evs.drive object if no drive is specified
    function noDrive(t)
        x=0
        return(x)
    end

    for i in 1:phdim
        evs.phononDrive[i]=noDrive
    end



    # FM initialization
    # expVals=zeros(dim^2,length(lattice))
    # for site in 1:length(lattice)
    #     if site==1
    #         vec=[0.0,0.2,1.0]
    #         vec/=norm(vec)
    #         push!(vec,1.0)
    #         expVals[:,site]=vec
    #     else
    #         vec=[0.0,0,1.0,1.0]
    #         expVals[:,site]=vec
    #     end
    # end

    # evs.lattice.expVals=deepcopy(expVals)
    # evs.latticePrev.expVals=deepcopy(expVals)

    # lattice.expVals=deepcopy(expVals)


    #AFM initialization
    # expVals=zeros(dim^2,length(lattice))
    # for site in 1:length(lattice)
    #     if site==1
    #         vec=[0.0,0.2,-1.0]
    #         vec/=norm(vec)
    #         push!(vec,1.0)
    #         expVals[:,site]=vec
    #     else
    #         row=floor((site-1)/lattice.size[1])
    #         vec=[0.0,0,1.0,1.0]
    #         if site%2==0
    #             sign=-1
    #         else
    #             sign=1
    #         end

    #         if row%2==0
    #             sign*=1
    #         else
    #             sign*=-1
    #         end

    #         expVals[:,site]=sign*vec
    #     end
    # end

    # evs.lattice.expVals=deepcopy(expVals)
    # evs.latticePrev.expVals=deepcopy(expVals)

    # lattice.expVals=deepcopy(expVals)



        


    return evs
end





