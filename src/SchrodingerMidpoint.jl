using DifferentialEquations
using Plots
#Constants and setup
mutable struct SchrodingerMP
    Zi::Matrix{ComplexF64}
    Zf1::Matrix{ComplexF64}
    Zf2::Matrix{ComplexF64}
    Zm::Matrix{ComplexF64}

    Qi::Matrix{Float64}
    Qf1::Matrix{Float64}
    Qf2::Matrix{Float64}
    Qm::Matrix{Float64}

    Pi::Matrix{Float64}
    Pf1::Matrix{Float64}
    Pf2::Matrix{Float64}
    Pm::Matrix{Float64}


    lattice::Lattice
    HZm::Vector{Matrix{ComplexF64}}
    phononMomenta::Matrix{Float64}
    phononMass::Vector{Float64}
    phononDamp::Vector{Float64}
    phononDrive::Vector{Function}
    dt::Float64
    timeStep::Tuple
    
    obs::EvolveObservables
    SchrodingerMP()=new{}()
end
function initSMP(dim,lattice,phdim)
    smp=SchrodingerMP()
    smp.lattice=deepcopy(lattice)
    smp.Zi = deepcopy(smp.lattice.spins)
    smp.Zf1 = Array{ComplexF64,2}(undef, dim, lattice.length)
    smp.Zf2 = Array{ComplexF64,2}(undef, dim, lattice.length)
    smp.Zm = Array{ComplexF64,2}(undef, dim, lattice.length)
    smp.HZm = Vector{Matrix{ComplexF64}}(undef, lattice.length) 


    smp.Qi = deepcopy(smp.lattice.phonons)
    smp.Qf1 = Array{Float64,2}(undef, phdim, lattice.length)
    smp.Qf2 = Array{Float64,2}(undef, phdim, lattice.length)
    smp.Qm = Array{Float64,2}(undef, phdim, lattice.length)


    smp.Pi = Array{Float64,2}(undef, phdim, lattice.length)
    smp.Pf1 = Array{Float64,2}(undef, phdim, lattice.length)
    smp.Pf2 = Array{Float64,2}(undef, phdim, lattice.length)
    smp.Pm = Array{Float64,2}(undef, phdim, lattice.length)




    smp.phononMass = zeros(phdim)
    smp.phononDamp = zeros(phdim)
    smp.phononDrive = Vector{Function}(undef,phdim)
    smp.phononMomenta = Array{Float64,2}(undef, phdim, lattice.length)
    smp.dt = 0.05
    smp.timeStep=(0,smp.dt)
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



function setQi!(smp,site,newState)
    smp.Qi[:,site] = newState
end
function getQi(smp,site)
    return(smp.Qi[:,site])
end
function setQf1!(smp,site,newState)
    smp.Qf1[:,site] = newState
end
function getQf1(smp,site)
    return(smp.Qf1[:,site])
end
function setQm!(smp,site,newState)
    smp.Qm[:,site] = newState
end
function getQm(smp,site)
    return(smp.Qm[:,site])
end



function setPi!(smp,site,newState)
    smp.Pi[:,site] = newState
end
function getPi(smp,site)
    return(smp.Pi[:,site])
end
function setPf1!(smp,site,newState)
    smp.Pf1[:,site] = newState
end
function getPf1(smp,site)
    return(smp.Pf1[:,site])
end
function setPm!(smp,site,newState)
    smp.Pm[:,site] = newState
end
function getPm(smp,site)
    return(smp.Pm[:,site])
end



function updateHZm!(smp, lattice, gens)
    for site in 1:length(lattice)
        p0=getQm(smp,site)
        pInteraction=smp.lattice.phononCoupling
        pRes=pInteraction*p0


        interactionSites = getInteractionSites(lattice, site)
        interactionMatrices = getInteractionMatrices(lattice, site)
        scale=zeros(ComplexF64,gens.dim^2)
        for i in 1:length(interactionSites)
            s1 = genExpVals(getZm(smp, interactionSites[i]), gens)
            Jex=interactionMatrices[i].mat
            scale+=Jex*s1
        end
        smp.HZm[site]=sum(scale.*gens.generators)+sum(pRes.*gens.generators)
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


function fullEvolve!(smp,lattice,gens,phdim)
    for site in 1:length(lattice)
        Z0 = getZi(smp,site)
        Zf = getZf1(smp,site)
        Zm = 0.5*(Zf + Z0)
        setZm!(smp,site,Zm)

        Q0 = getQi(smp,site)
        Qf = getQf1(smp,site)
        Qm=0.5*(Qf+Q0)
        setQm!(smp,site,Qm)


        P0 = getPi(smp,site)
        Pf = getPf1(smp,site)
        Pm=0.5*(Pf+P0)
        setPm!(smp,site,Pm)
    end
    updateHZm!(smp,lattice,gens)
    for site in 1:length(lattice)
        Z0 = getZi(smp,site)
        Zm = getZm(smp,site)
        Zf = Z0 - ((1im*smp.dt)*(smp.HZm[site]*Zm))
        setZf1!(smp,site,Zf)

        # if site==1
        #     print(getZf1(smp,site),"\n")
        # end

        setZf2!(smp,site,Zf)
        


        P0 = getPi(smp,site)
        Pm = getPm(smp,site)


        Q0 = getQi(smp,site)
        Qm = getQm(smp,site)

        coupling = smp.lattice.phononCoupling
        Sm=midpointExpVal(gens,getZm(smp,site))

        Qf = Q0 + smp.dt*Pm./smp.phononMass-smp.dt*smp.phononDamp.*Qm
        Pf = P0 - smp.dt*smp.lattice.springConstants.*Qm - smp.dt*transpose(coupling)*Sm
        setQf1!(smp,site,Qf)
        setPf1!(smp,site,Pf)
    end


end


function midpointExpVal(gens,s1)
    vals=zeros(ComplexF64, gens.dim^2)
    for i in 1:length(gens.generators)
        vals[i]=calcInnerProd(s1,gens.generators[i],s1)
    end
    return(vals)
end











# function evolvePhononSMP(gens,smp,site,dim,phdim)
#     x0=getPhonon(smp.lattice,site)
#     p0=getPhononMomentum(smp,site)


#     append!(x0,p0)
  

#     s0= getExpValSpin(smp,site)

#     coupling = smp.lattice.phononCoupling
#     springConst = smp.lattice.springConstants
#     damping = smp.phononDamp



#     vec=transpose(coupling)*s0



#     function update(xdot,x,p,t)
#         xdot[1:phdim] = (x[phdim+1:end]./(smp.phononMass))-damping.*x[1:phdim]
#         xdot[phdim+1:end] = -springConst.*x[1:phdim]-vec
#         for i in 1:phdim
#             xdot[i+phdim] += smp.phononDrive[i](t)
#         end


#     end

#     alg = RK4()
#     phononProb = ODEProblem(update,x0,smp.timeStep)
#     sol = solve(phononProb, alg)

#     return (last(sol.u))
# end



function evolveSMP!(smp,lattice,gens,phdim)
    # set Zf1 to Z0 for first iteration
    finalState!(smp.lattice,gens)
    for site in 1:length(lattice)
        Z0 = getZi(smp,site)
        Q0 = getQi(smp,site)
        P0=getPi(smp,site)

        setZf1!(smp,site,Z0)
        setZf2!(smp,site,Z0)
        setQf1!(smp,site,Q0)
        setPf1!(smp,site,P0)
    end
    iterations=10
    for i in 1:iterations
        Q1=getQf1(smp,1)[1]
        fullEvolve!(smp,lattice,gens,phdim)
        Q2=getQf1(smp,1)[1]
        # if(i==10)
        #     print(abs(Q2-Q1),"\n")
        # end
    end
    for site in 1:length(lattice)
        # phonon=evolvePhononSMP(gens,smp,site,gens.dim,phdim)
        # setPhonon!(smp.lattice,site,phonon[1:phdim])
        # setPhononMomentum!(smp,site,phonon[phdim+1:end])


        Z = getZf1(smp,site)
        setZi!(smp,site,Z)
        setSpin!(lattice, site, Z)

        Q = getQf1(smp,site)
        setQi!(smp,site,Q)
        setPhonon!(lattice,site,Q)


        P = getPf1(smp,site)
        setPi!(smp,site,P)
        setPhononMomentum!(smp,site,P)
    end
end

function initPhMomentum!(smp,T,phd)

    Umax=0.5*smp.lattice.springConstants.*(smp.lattice.Qmax.^2)
    bound=exp.(-Umax/T)
    for site in 1:length(smp.lattice)
        P=zeros(phd)
        p0=getPhonon(smp.lattice,site)
        Umin=0.5*smp.lattice.springConstants.*(p0.^2)
        LowBound=exp.(-Umin/T)
        for ph in 1:phd
            P[ph]=rand(Uniform(bound[ph],LowBound[ph]))
        end
        
        
        A=-2*T*log.(P)
        B=smp.lattice.springConstants.*(p0.^2)
        v=sqrt.((A .- B).*smp.phononMass)

        choice=[1.0,-1.0]
        sign=rand(choice,phd)

        v.*=sign


        setPhononMomentum!(smp,site,v)
    end

    smp.Pi=deepcopy(smp.phononMomenta)
end


function updateTimeSpan!(smp,stepSize)
    smp.timeStep = (smp.timeStep[2], smp.timeStep[2]+(stepSize))
end

function setTimeStep!(smp)
    smp.timeStep=smp.timeStep[2]-smp.timeStep[1]
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

function getExpValSpin(smp, site)
    return smp.lattice.expVals[:,site]
end