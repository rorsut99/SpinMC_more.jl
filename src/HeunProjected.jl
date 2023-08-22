using DifferentialEquations
mutable struct HeunP
    Z1::Matrix{ComplexF64}
    Z2::Matrix{ComplexF64}
    lattice::Lattice
    HZ::Vector{Matrix{ComplexF64}}
    HZ1::Vector{Matrix{ComplexF64}}

    phononMomenta::Matrix{Float64}
    phononMass::Vector{Float64}
    phononDamp::Vector{Float64}
    phononDrive::Vector{Function}
    timeStep::Tuple
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
    hp.dt = 0.05
    hp.obs=initEvolveObservables()
    hp.timeStep=(0,hp.dt)

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

    Umax=0.5*hp.lattice.springConstants.*(hp.lattice.Qmax.^2)
    bound=exp.(-Umax/T)
    for site in 1:length(hp.lattice)
        P=zeros(phd)
        p0=getPhonon(hp.lattice,site)
        Umin=0.5*hp.lattice.springConstants.*(p0.^2)
        LowBound=exp.(-Umin/T)
        for ph in 1:phd
            P[ph]=rand(Uniform(bound[ph],LowBound[ph]))
        end
        
        
        A=-2*T*log.(P)
        B=hp.lattice.springConstants.*(p0.^2)
        v=sqrt.((A .- B).*hp.phononMass)

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
    hp.timeStep=(0,dt)
end

function updateHZ!(hp, gens)

    for site in 1:length(hp.lattice)
        p0=getPhonon(hp.lattice,site)
        pInteraction=hp.lattice.phononCoupling
        pRes=pInteraction*p0

        interactionSites = getInteractionSites(hp.lattice, site)
        interactionMatrices = getInteractionMatrices(hp.lattice, site)
        scale=zeros(ComplexF64,gens.dim^2)
        for i in 1:length(interactionSites)
            s1 = genExpVals(getSpin(hp.lattice, interactionSites[i]), gens)
            Jex=interactionMatrices[i].mat
            scale+=Jex*s1
        end
        hp.HZ[site]=sum(scale.*gens.generators)+sum(pRes.*gens.generators)
    end
end

function updateHZ1!(hp, gens)

    for site in 1:length(hp.lattice)
        p0=getPhonon(hp.lattice,site)
        pInteraction=hp.lattice.phononCoupling
        pRes=pInteraction*p0

        interactionSites = getInteractionSites(hp.lattice, site)
        interactionMatrices = getInteractionMatrices(hp.lattice, site)
        scale=zeros(ComplexF64,gens.dim^2)
        for i in 1:length(interactionSites)
            s1 = genExpVals(getZ1!(hp, interactionSites[i]), gens)
            Jex=interactionMatrices[i].mat
            scale+=Jex*s1
        end
        hp.HZ1[site]=sum(scale.*gens.generators)+sum(pRes.*gens.generators)
    end
end

function evolveSpinHP!(hp,gens)
    updateHZ!(hp, gens)
    for site in 1:length(hp.lattice)
        Z0=getSpin(hp.lattice,site)
        newState=Z0-1im*hp.dt*hp.HZ[site]*Z0
        newState/=norm(newState)
        newState*=sqrt(2)
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

function evolve_phononHP(gens,hp,site,dim,phdim)
    x0=getPhonon(hp.lattice,site)
    p0=getPhononMomentum(hp,site)


    append!(x0,p0)
  

    s0= getExpValSpin(hp,site)

    coupling = hp.lattice.phononCoupling
    springConst = hp.lattice.springConstants
    damping = hp.phononDamp



    vec=transpose(coupling)*s0



    function update(xdot,x,p,t)
        xdot[1:phdim] = (x[phdim+1:end]./(hp.phononMass))-damping.*x[1:phdim]
        xdot[phdim+1:end] = -springConst.*x[1:phdim]-vec
        for i in 1:phdim
            xdot[i+phdim] += hp.phononDrive[i](t)
        end


    end

    alg = RK4()
    phononProb = ODEProblem(update,x0,hp.timeStep)
    sol = solve(phononProb, alg)

    return (last(sol.u))
end


function getExpValSpin(hp, site)
    return hp.lattice.expVals[:,site]
end


function addPhononDrive!(hp,driveFunctions,phd)
    length(driveFunctions) == (phd) || error(string("Phonon drive functions must be of size ",phd,"."))
    hp.phononDrive = driveFunctions
end

function updateTimeSpan!(hp,stepSize)
    hp.timeStep = (hp.timeStep[2], hp.timeStep[2]+(stepSize))
end

function setTimeStep!(hp)
    hp.timeStep=hp.timeStep[2]-hp.timeStep[1]
end

function updateTimeStep!(hp, tstep)
    hp.timeStep = tstep
end

function setPhononMass!(hp,vec,phd)
    length(vec) == (phd) || error(string("Phonon mass must be of size ",phd,"."))
    hp.phononMass = vec
end

function setPhononDamp!(hp,vec,phd)
    length(vec) == (phd) || error(string("Phonon damping constants must be of size ",phd,"."))
    hp.phononDamp = vec
end





function evolveHP!(hp,gens,phdim)
    evolveSpinHP!(hp,gens)
    finalState!(hp.lattice,gens)
    for site in 1:length(hp.lattice)
        phonon=evolve_phononHP(gens,hp,site,gens.dim,phdim)
        setPhonon!(hp.lattice,site,phonon[1:phdim])
        setPhononMomentum!(hp,site,phonon[phdim+1:end])



        Z2=getZ2!(hp,site)
        Z2/=norm(Z2)
        Z2*=sqrt(2)
        setSpin!(hp.lattice,site,Z2)
    end
end
    









