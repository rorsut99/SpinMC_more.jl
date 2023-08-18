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
# include("evolve.jl")

using LinearAlgebra
using Suppressor
using Plots
using MPI

function makeGenerators(dim::Int)
    # produce generator matrices for different dim
    if (dim==2)
        # Pauli matrices
        sx=[0+0.0im 1.0+0im 
            1.0+0im 0+0im]
        sy=[0 -1.0im
            1.0im 0]
        sz=[1.0+0im 0
            0 -1.0+0im]
        s9 = Matrix(1.0I, 2, 2)

        # calculate norm
        norm = sx^2 + sy^2 + sz^2 + s9^2

        generators = [sx, sy, sz, s9]

    elseif (dim==3)
        # Gell-Mann matrices
        sx=[0+0.0im 1.0+0im  0
            1.0+0im 0+0im 0
            0 0 0]
        sy=[0 -1.0im 0
            1.0im 0 0
            0 0 0 ]
        sz=[1.0+0im 0 0
            0 -1.0+0im 0
            0 0 0  ]
        s4=[0+0im 0 1.0
            0 0 0
            1.0 0 0]
        s5=[0 0 -1.0im
            0 0 0
            1.0im 0 0]
        s6=[0+0.0im 0 0
            0 0 1.0
            0 1.0 0]
        s7=[0 0 0
            0 0 -1.0im
            0 1.0im 0]
        s8=[1.0+0im 0 0
            0 1.0 0
            0 0 -2.0]/sqrt(3)
        s9 = Matrix(1.0I, 3, 3)

        # calculate norm
        norm = sx^2 + sy^2 + sz^2 + s4^2 + s5^2 + s6^2 + s7^2 + s8^2 + s9^2
        print(norm)
     
        generators = [sx, sy, sz, s4, s5, s6, s7, s8, s9]
    end

    return generators
end

function makeLattice(dim::Int, dim2::Int, phdim::Int)

    # define cubic lattice with Heisenberg interaction
    a1=(sqrt(3)/2,1.0/2)
    a2=(sqrt(3)/2,-1.0/2)

    gens=initGen(dim)

    # Sx=[0+0.0im 1.0+0im 
    # 1.0+0im 0+0im]
    # Sy=[0 -1.0im
    # 1.0im 0]
    # Sz=[1.0+0im 0
    # 0 -1.0+0im]

    # Sx=[0 0 0.0+0im
    # 0 0 -1im
    # 0 1im 0]

    # Sy=[0 0 0.0+1.0im
    # 0 0 0
    # -1im 0 0]

    # Sz=[0.0+0im -1im 0
    # 1im 0 0
    # 0 0 0]

    Sx=[0 0 1.0+0im
    0 0 1
    1 1 0]/sqrt(2)

    Sy=(-1im/sqrt(2))*[0 0 1.0+0im
    0 0 -1
    -1 1 0]

    Sz=[1.0+0im 0 0
    0 -1 0
    0 0 0]

    addSpinOperator!(gens,Sx)
    addSpinOperator!(gens,Sy)
    addSpinOperator!(gens,Sz)
 

    generators = makeGenerators(dim)

    # push generators to lattice struct
    for gen in generators
        addGenerator!(gens,gen)
    end
    setGenReps!(gens)


    uc = UnitCell(a1,a2)
    FMint = Matrix(-1.0I,3,3)      # Heisenberg interaction
    AFMint = Matrix(1.0I,3,3)      # Heisenberg interaction
    Zero = Matrix(0.0I,dim,dim)
    addBasisSite!(uc,(0.0,0.0),dim)
    # nearest neighbour interaction
    # added parameter to take in the order of the term in the hamiltonian
    # bilinear interaction
    J1 = 1
    addInteraction!(uc,gens,1,1,J1*AFMint,1,(1,0))
    addInteraction!(uc,gens,1,1,J1*AFMint,1,(0,1))
    addInteraction!(uc,gens,1,1,J1*AFMint,1,(1,-1))

    B=[0.0,0,0]
    setField!(uc,gens,1,B)

    # # biquadratic interaction
    J2 = 1.5
    addInteraction!(uc,gens,1,1,J2*FMint,2,(1,0))
    addInteraction!(uc,gens,1,1,J2*FMint,2,(0,1))
    addInteraction!(uc,gens,1,1,J2*FMint,2,(1,-1))
 



    

    Lsize=(16,16)       # size of lattice
    lattice=Lattice(uc,Lsize,dim,phdim)

    

    spring = [0.0,0.0]
    # mat = [1.0; 0.0; 0.0 ;;]
    mat =[0.0 0.0
        0.0  0.0
        0.0  0.0]

    addSpringConstant!(lattice, spring, phdim)
    addPhononInteraction!(lattice,1, gens, mat)


    
    # # touch("mc_plots/output_BBQTriLattice_J1-1_J2-15.txt")
    # file = open("mc_plots/output_BBQTriLattice_SU2.txt", "w")

    # write(file, string("\n\nResults for MC simulation on SU(", dim, ") BBQ Spin-1 Triangular Lattice\n\n"))
    # write(file, string("MODEL PARAMETERS--------------------------------\nlattice size: ", Lsize, "\n"))
    # write(file, string("Bilinear J1: ", J1, "\n"))
    # write(file, string("Biquadratic J2: ", J2, "\n"))
    # write(file, string("Spin Operators: \nSx: ", Sx, "\nSy: ", Sy, "\nSz: ", Sz, "\n"))
    # write(file, string("Number of phonons: ", phdim, "\n"))
    # write(file, string("Spring constants: ", spring, "\nCoupling: ", mat, "\n\n"))

    

    # Sx=[0 0 1.0+0im
    # 0 0 1
    # 1 1 0]/sqrt(2)

    # Sy=(-1im/sqrt(2))*[0 0 1.0+0im
    # 0 0 -1
    # -1 1 0]

    # Sz=[1.0+0im 0 0
    # 0 -1 0
    # 0 0 0]

    # return (lattice,gens,file)
    return (lattice,gens)
end

function runMC(T)
    # define dimensions
    dim=3           # dimension of wavefunction (N)
    dim2=dim^2-1    # dimension of spin vector (N^2-1)

    phdim=2

    # MPI.Initialized() || MPI.Init()
    # commSize = MPI.Comm_size(MPI.COMM_WORLD)
    # commRank = MPI.Comm_rank(MPI.COMM_WORLD)

    # set sweeps
    thermSweeps=200
    sampleSweeps=200

    # # temp=ones(length(T))
    # tmin=0.1
    # tmax=0.7
    # T=LinRange(tmin, tmax, commSize)[commRank+1]

    beta=1.0/T
    #beta=LinRange(1.0/tmax, 1.0/tmin, commSize)[commRank+1]
    # beta=1/T
    # beta = (commSize == 1) ? 1.0/tmin : 1.0 / (reverse([ tmax * (tmin / tmax)^(n/(commSize-1)) for n in 0:commSize-1 ])[commRank+1])
    # beta = temp./T
    # T = 1000.0
    # lattice,gens,file = makeLattice(dim, dim2, phdim)
    lattice,gens = makeLattice(dim, dim2, phdim)
    lattice.Qmax = 0.05

    # run Monte Carlo sweeps
    m=MonteCarlo(lattice,beta,thermSweeps,sampleSweeps,dim,replicaExchangeRate=10)
    run!(m,gens,dim, phdim)
    e,e2=means(m.observables.energy)

    # write(file, string("MC PARAMETERS-----------------------------------\n"))
    # write(file, string("temperature: ", 1/beta, "\n"))
    # write(file, string("number of therm sweeps: ", thermSweeps, "\n"))
    # write(file, string("number of measure sweeps: ", sampleSweeps, "\n\n"))

    # # # print magnetization
    # # print("Magnetization vector: ", getMagnetization(m.lattice,dim), "\n")

    # # # print energy
    # write(file, string("MC RESULTS--------------------------------------\n"))
    # write(file, string("Final energy: ", e, "\nFinal energy squared: ", e2, "\n"))
    # close(file)

    # print(m.lattice.spins)
    # return (m, gens, file)
    # return (m.energySeries)
    return(m,gens)
    # c(e) = beta * beta * (e[2] - e[1] * e[1]) * length(m.lattice) 
    # return mean(m.observables.energy, c)
end

# Tvals = LinRange(0.1, 4, 40)
# heat = zeros(40)
# for i in 1:length(Tvals)
#     heat[i] = runMC(Tvals[i])
# end


# tick= time()
# heat = runMC(0.001)
# tock = time()

#  # plot energy vs sweeps
# title = string("SU(", dim, ") FM heat capacity")
# plot(Tvals, heat, title=title)
# xlabel!("T")
# ylabel!("C")


# Tpoints=30
# Tvals = LinRange(0.1, 0.7, Tpoints)
# heat = zeros(Tpoints)

# tick = time()
m,gens=runMC(0.001)
# tock = time()
# print(tock-tick)

# file = open("mc_plots/output_BBQTriLattice_FM.txt", "w")
# write(file, string("MC runtime: ", tock-tick, "seconds\n"))
# close(file)


# for i in 1:length(Tvals)
#     heat[i] = runMC(Tvals[i])
# end


# title = string("SU(", dim, ") FM heat capacity")
# plot(Tvals, heat, title=title)
# xlabel!("T")
# ylabel!("C")

# dim=2        # dimension of wavefunction (N)
# dim2=dim^2-1    # dimension of spin vector (N^2-1)

# phdim=4

# lattice,generators=makeLattice(dim,dim2,phdim)

# m=runMC(0.01)
# print(m.energySeries)
# # plot energy vs sweeps
# title = string("SU(", dim, ") FM interaction")
# plot(m.energySeries, title=title)
# xlabel!("sweeps")
# ylabel!("energy density")