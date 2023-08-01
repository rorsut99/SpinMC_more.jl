using DifferentialEquations
using Plots
using SciPyDiffEq

#Constants and setup


mutable struct Evolution
    structureFactors::Matrix{Vector{ComplexF64}}
    latticePrev::Lattice
    lattice::Lattice
    phononMomentaPrev::Matrix{Float64}
    phononMomenta::Matrix{Float64}
    phononMass::Vector{Float64}
    phononDamp::Vector{Float64}
    phononDrive::Vector{Function}
    timeStep::Float64
    tspan::Tuple
    threshold::Float64

    obs::EvolveObservables

    Evolution()=new{}()
end


function initEv(dim,lattice,gens,timeStep,phdim)
    evs = Evolution()
    finalState!(lattice,gens,dim) # save the final MC state to lattice.expVals
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


        


    return evs
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
            evs.structureFactors[i,j]=-1im*vec

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

    # if site==1
    #     print("output: ", output,"\n")
    #     mat=zeros(dim^2,dim^2)
    #     for i in 1:dim^2
    #         for j in 1:dim^2
    #             mat[i,j]=dot(evs.structureFactors[i,j], s0)
    #         end
    #     end
    #     print("mat*output: ", mat*output, "\n")
    # end


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

    alg = Tsit5()
    spin_prob = ODEProblem(update,s0,evs.tspan)
    sol = solve(spin_prob, alg)



    return (last(sol.u))

end

function evolve_phonon(gens,evs,site,dim,phdim)
    x0=getPhonon(evs.latticePrev,site)
    p0=getPhononMomentum(evs,site)

    append!(x0,p0)

    s0= getExpValSpin(evs,site)

    coupling = evs.lattice.phononCoupling
    springConst = evs.lattice.springConstants
    damping = evs.phononDamp



    vec=transpose(coupling)*s0

    

    # function update(xdot,x,p,t)
    #     alpha=damping./(evs.phononMass)
    #     xdot[1:phdim] = -(x[phdim+1:end]./(evs.phononMass)).*exp.(-alpha*t)
    #     xdot[phdim+1:end] = springConst.*x[1:phdim].*exp.(alpha*t)+vec.*exp.(alpha*t/2)

    #     for i in 1:phdim
    #         xdot[i+phdim] -= evs.phononDrive[i](t)*exp(alpha[i]*t/2)
    #     end
    # end


    function update(xdot,x,p,t)
        xdot[1:phdim] = (x[phdim+1:end]./(evs.phononMass))-damping.*x[1:phdim]
        xdot[phdim+1:end] = -springConst.*x[1:phdim]-vec
        for i in 1:phdim
            xdot[i+phdim] += evs.phononDrive[i](t)
        end


    end

    alg = SciPyDiffEq.RK45()
    phononProb = ODEProblem(update,x0,evs.tspan)
    sol = solve(phononProb, alg)

    return (last(sol.u))
end

function addPhononDrive!(evs,driveFunctions,phd)
    length(driveFunctions) == (phd) || error(string("Phonon drive functions must be of size ",phd,"."))
    evs.phononDrive = driveFunctions
end

function updateTimeSpan!(evs,stepSize)
    evs.tspan = (evs.tspan[2], evs.tspan[2]+(stepSize))
end

function setTimeStep!(evs)
    evs.timeStep=evs.tspan[2]-evs.tspan[1]
end

function updateTimeStep!(evs, tstep)
    evs.timeStep = tstep
end

function setThreshold!(evs, threshold)
    evs.threshold = threshold
end


# function checkStability(evs)
#     spinEnergy, phEnergy, totalEnergyPrev = getEvEnergy(evs,gens,evs.latticePrev)
#     spinEnergy, phEnergy, totalEnergy = getEvEnergy(evs,gens,evs.lattice)
#     diff = totalEnergy - totalEnergyPrev
# end




function evolve!(evs,gens,dim,phdim,T,numSteps)

    for site in 1:length(evs.lattice)
        spin=evolve_spin(gens,evs,site,dim)
        # phonon=evolve_phonon(gens,evs,site,dim,phdim)
        setExpValSpin!(evs,site,spin)
        # setPhonon!(evs.lattice,site,phonon[1:phdim])
        # setPhononMomentum!(evs,site,phonon[phdim+1:end])
            
    end
    # check stability, add if statment
    # evs.phononMomentaPrev = deepcopy(evs.phononMomenta)
    evs.latticePrev = deepcopy(evs.lattice)
    updateTimeSpan!(evs,evs.timeStep)
    # print(i, "\n")






end