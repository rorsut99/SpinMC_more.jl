include("Generators.jl")
include("UnitCell.jl")
include("InteractionMatrix.jl")
include("Lattice.jl")
include("Spin.jl")
include("Observables.jl")
include("MonteCarlo.jl")
include("Helper.jl")
include("IO.jl")
include("Phonon.jl")
include("EvolveObservables.jl")
include("SchrodingerMidpoint.jl")

using HDF5
using Serialization
using JLD2




function parseData()
    K=["0.5","0.75","1.00","1.25","1.50","1.75","2.00","2.25","2.50","2.75","3.00","3.25","3.50","3.75","4.0","4.25","4.50","4.75","5.00"]
    stem="strongQuadMCdatK="
    file="pseudoSpin--strongQuad.h5.."

    gens=initGen(2)

    Sx=0.5*[0+0.0im 1.0+0im
            1.0+0im 0+0im]
    Sy=0.5*[0 -1.0im
            1.0im 0]
    Sz=0.5*[1.0+0im 0
            0 -1.0+0im]
    Id=0.5*Matrix((1.0+0.0im)I, 2, 2)

    addSpinOperator!(gens,Sx)
    addSpinOperator!(gens,Sy)
    addSpinOperator!(gens,Sz)

    addGenerator!(gens,Sx)
    addGenerator!(gens,Sy)
    addGenerator!(gens,Sz)
    addGenerator!(gens,Id)

    setGenReps!(gens)







    nK=19
    nFiles=80
    peaks=zeros(nK)
    Mx=zeros(nK,nFiles)
    My=zeros(nK,nFiles)
    Mz=zeros(nK,nFiles)
    aMx=zeros(nK,nFiles)
    aMy=zeros(nK,nFiles)
    aMz=zeros(nK,nFiles)
    for j in 1:nK
        heat=zeros(nFiles)
        T=zeros(nFiles)
        k=K[j]
        for i in 0:nFiles-1
            filename=string(stem,k,"/",file,i)

            m = readMonteCarlo(filename)
            finalState!(m.lattice,gens)
            beta=m.beta


            c(e) = beta * beta * (e[2] - e[1] * e[1]) * length(m.lattice)
            ∇c(e) = [-2.0 * beta * beta * e[1] * length(m.lattice), beta * beta * length(m.lattice)]
            heat[i+1] = mean(m.observables.energy, c)
            T[i+1]=1.0/beta
            


            
            Mx[j,i] = mean((m.lattice.expVals[1,:]))
            My[j,i] = mean((m.lattice.expVals[2,:]))
            Mz[j,i] = mean((m.lattice.expVals[3,:]))

            aMx[j,i] = mean(abs.(m.lattice.expVals[1,:]))
            aMy[j,i] = mean(abs.(m.lattice.expVals[2,:]))
            aMz[j,i] = mean(abs.(m.lattice.expVals[3,:]))
        end


        peaks[j]=T[findmax(heat)[2]]
    end


    return(peaks,Mx,My,Mz,aMx,aMy,aMz)
end


peak,Mx,My,Mz,aMx,aMy,aMz=parseData()
outfile="pseudoSpinPhaseDiagram.h5"

jldsave(outfile;peak,Mx,My,Mz,aMx,aMy,aMz)