using DifferentialEquations
using Plots

#Constants and setup


mutable struct Evolution
    structureFactors::Matrix{Vector{ComplexF64}}
    latticePrev::Lattice
    lattice::Lattice
    tspan::Tuple

    Evolution()=new{}()
end


function initEv(dim,lattice,timeStep)
    evs = Evolution()
    evs.structureFactors = Matrix{Vector{ComplexF64}}(undef,dim^2,dim^2)
    evs.lattice = lattice
    evs.latticePrev = lattice
    evs.tspan = timeStep
    return evs
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


function evolve_spins(gens,evs,site,dim)
    Id=Matrix((1.0+0im)I,dim,dim)

    s0=genExpVals(getSpin(evs.latticePrev, site), gens,dim)
    tempS0=copy(s0)
    push!(tempS0,calcInnerProd(getSpin(evs.latticePrev,site),Id,getSpin(evs.latticePrev,site)))
    

    interactionSites = getInteractionSites(evs.latticePrev, site)
    interactionMatrices = getInteractionMatrices(evs.latticePrev, site)
    output=zeros(dim^2)
    for j in 1:length(interactionSites)
        Jex=interactionMatrices[j].mat
        s1 = genExpVals(getSpin(evs.latticePrev, interactionSites[j]), gens,dim)
        tempS1=copy(s1)
        push!(tempS1,calcInnerProd(getSpin(evs.latticePrev,interactionSites[j]),Id,getSpin(evs.latticePrev,interactionSites[j])))


        output+=Jex*tempS1
    end


    
  



    function update(xdot,x,p,t)
        mat=zeros(dim^2,dim^2)
        for i in 1:dim^2
            for j in 1:dim^2
                mat[i,j]=sum((evs.structureFactors.*x)[i,j])
            end
        end
        

        xdot[:] = mat*output


    end


    spin_prob = ODEProblem(update,tempS0,evs.tspan)
    sol = solve(spin_prob)



    return (last(sol.u))

end



function solve_oscillator(pos,springConst)
    #Define the Problem

    initial=[pos,0]
    function oscillator(xdot, x, p, t)
        xdot[1] = x[2]
        xdot[2] = -springConst*x[1]
    end

    #Pass to Solvers
    harmonic_oscillator_problem = ODEProblem(oscillator, initial, tspan)
    sol = solve(harmonic_oscillator_problem)

    return sol
end

function evolve_oscillators(lattice)
    res=0
    for i in 1:length(lattice)
        for j in 1:length(lattice.phonons[:,i])
            res=solve_oscillator(lattice.phonons[:,i][j],lattice.springConstants[j])
            
        end
    end
    plot(res)
    finalPos=last(res.u)[1]
    finalV=last(res.u)[2]
    print(res.u)

end