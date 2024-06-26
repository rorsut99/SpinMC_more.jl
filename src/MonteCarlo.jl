using Random
using Dates
using Printf
using MPI

mutable struct MonteCarloStatistics
    sweeps::Int
    attemptedLocalUpdates::Int
    acceptedLocalUpdates::Int
    attemptedReplicaExchanges::Int
    acceptedReplicaExchanges::Int
    initializationTime::Float64

    MonteCarloStatistics() = new(0, 0, 0, 0, 0, time())
end

mutable struct MonteCarlo{T<:Lattice,U<:AbstractRNG}
    lattice::T
    
    beta::Float64
    thermalizationSweeps::Int
    measurementSweeps::Int
    measurementRate::Int
    replicaExchangeRate::Int
    reportInterval::Int
    checkpointInterval::Int

    rng::U
    seed::UInt
    sweep::Int




    energySeries::Vector{Float64}
    observables::Observables
end

function MonteCarlo(
    lattice::T, 
    beta::Float64, 
    thermalizationSweeps::Int, 
    measurementSweeps::Int,dim::Int; 
    measurementRate::Int = 1, 
    replicaExchangeRate::Int = 10, 
    reportInterval::Int = round(Int, 0.05 * (thermalizationSweeps + measurementSweeps)), 
    checkpointInterval::Int = 3600, 
    rng::U = copy(Random.GLOBAL_RNG), 
    seed::UInt = rand(Random.RandomDevice(),UInt)
    ) where T<:Lattice where U<:AbstractRNG

    mc = MonteCarlo(deepcopy(lattice), beta, thermalizationSweeps, measurementSweeps, measurementRate, replicaExchangeRate, reportInterval, checkpointInterval, rng, seed, 0,zeros(thermalizationSweeps+measurementSweeps), Observables(lattice,dim))
    Random.seed!(mc.rng, mc.seed)
    
    return mc
end

function run!(mc::MonteCarlo{T},gens::Generators, dim::Int, phdim::Int; outfile::Union{String,Nothing}=nothing, samplesOutfile::Union{String,Nothing}=nothing) where T<:Lattice
    #init MPI
    rank = 0
    commSize = 1
    allBetas = zeros(0)
    enableMPI = false
    if MPI.Initialized()
        commSize = MPI.Comm_size(MPI.COMM_WORLD)
        rank = MPI.Comm_rank(MPI.COMM_WORLD)
        if commSize > 1
            allBetas = zeros(commSize)
            allBetas[rank + 1] = mc.beta
            MPI.Allgather!(UBuffer(allBetas, 1), MPI.COMM_WORLD)
            enableMPI = true
            rank == 0 && @printf("MPI detected. Enabling replica exchanges across %d simulations.\n", commSize)
        end
    end

    #init IO
    enableOutput = typeof(outfile) != Nothing
    if enableOutput
        enableMPI && (outfile *= "." * string(rank))
        isfile(outfile) && error("File ", outfile, " already exists. Terminating.")
    end


    enableSamples = typeof(samplesOutfile) != Nothing
    
    #init spin configuration
    if mc.sweep == 0
        for i in 1:length(mc.lattice)
            # dim is not defined locally
            setSpin!(mc.lattice, i, uniformOnSphere(dim))
            # setSpin!(mc.lattice, i, [0.0+0im,0.0,1.0])
            setPhonon!(mc.lattice, i, uniformDist(phdim, mc.lattice.Qmax))
        end
    end

    #init Monte Carlo run
    totalSweeps = mc.thermalizationSweeps + mc.measurementSweeps
    partnerSpinConfiguration = deepcopy(mc.lattice.spins)
    partnerPhononConfiguration = deepcopy(mc.lattice.phonons)
    energy = getEnergy(mc.lattice,gens)

    if(energy<0)
        print(energy,"\n")
    end



    #launch Monte Carlo run
    lastCheckpointTime = time()
    statistics = MonteCarloStatistics()
    rank == 0 && @printf("Simulation started on %s.\n\n", Dates.format(Dates.now(), "dd u yyyy HH:MM:SS"))

    while mc.sweep < totalSweeps
        # if(energy<0)
        #     print(energy,"\n")
        # end
        #perform local sweep
        for i in 1:length(mc.lattice)
            #select random spin
            site = rand(mc.rng, 1:length(mc.lattice))

            #propose new spin configuration
            newSpinState = proposeUpdate(site,mc.lattice,gens,dim)
            energyDifference = getSpinEnergyDifference(mc.lattice,gens, site, newSpinState)

            #check acceptance of new configuration
            statistics.attemptedLocalUpdates += 1
            p = exp(-mc.beta * energyDifference)
            if (rand(mc.rng) < min(1.0, p))
                setSpin!(mc.lattice, site, newSpinState)
                energy += energyDifference
                statistics.acceptedLocalUpdates += 1
            end
        end

        for i in 1:length(mc.lattice)
            #select random spin
            site = rand(mc.rng, 1:length(mc.lattice))

            #propose new spin configuration
            newPhState = uniformDist(phdim, mc.lattice.Qmax)
            energyDifference = getPhononEnergyDifference(mc.lattice, gens, site, newPhState)
            

            #check acceptance of new configuration
            statistics.attemptedLocalUpdates += 1
            p = exp(-mc.beta * energyDifference)
            if (rand(mc.rng) < min(1.0, p))
                setPhonon!(mc.lattice,site,newPhState)
                energy += energyDifference
                statistics.acceptedLocalUpdates += 1
            end
        end
        statistics.sweeps += 1


        #perform measurement
        if mc.sweep >= mc.thermalizationSweeps
            if mc.sweep % mc.measurementRate == 0
                performMeasurements!(mc.observables, mc.lattice, energy, gens, dim)
            end
        end

        spinEnergy, phononEnergy = getPTEnergy(mc.lattice,gens)

        #perform replica exchange
        if enableMPI && mc.sweep % mc.replicaExchangeRate == 0
            #determine MPI rank to exchange configuration with
            if iseven(mc.sweep ÷ mc.replicaExchangeRate)
                partnerRank = iseven(rank) ? rank + 1 : rank - 1
            else
                partnerRank = iseven(rank) ? rank - 1 : rank + 1
            end

            if partnerRank >= 0 && partnerRank < commSize
                #obtain energy of new configuration
                partnerSpinEnergy = MPISendrecvFloat(spinEnergy, partnerRank, MPI.COMM_WORLD)
                partnerPhononEnergy = MPISendrecvFloat(phononEnergy, partnerRank, MPI.COMM_WORLD)
                partnerEnergy = MPISendrecvFloat(energy, partnerRank, MPI.COMM_WORLD)


                #check acceptance of new configuration
                statistics.attemptedReplicaExchanges += 1
                exchangeAccepted = false
                if iseven(rank)
                    pSpin = exp(-(allBetas[rank + 1] - allBetas[partnerRank + 1]) * (partnerSpinEnergy - spinEnergy))
                    pPhonon = exp(-(allBetas[rank + 1] - allBetas[partnerRank + 1]) * (partnerPhononEnergy - phononEnergy))

                    spinAccepted = (rand(mc.rng) < min(1.0, pSpin)) ? true : false
                    exchangeAccepted = ((rand(mc.rng) < min(1.0, pPhonon)) && spinAccepted) ? true : false

                    MPISendBool(exchangeAccepted, partnerRank, MPI.COMM_WORLD)
                else
                    exchangeAccepted = MPIRecvBool(partnerRank, MPI.COMM_WORLD)
                end
                if (exchangeAccepted)
                    energy = partnerEnergy
                    MPI.Sendrecv!(mc.lattice.spins, partnerRank, 0, partnerSpinConfiguration, partnerRank, 0, MPI.COMM_WORLD)
                    MPI.Sendrecv!(mc.lattice.phonons, partnerRank, 0, partnerPhononConfiguration, partnerRank, 0, MPI.COMM_WORLD)
                    (mc.lattice.spins, partnerSpinConfiguration) = (partnerSpinConfiguration, mc.lattice.spins)
                    (mc.lattice.phonons, partnerPhononConfiguration) = (partnerPhononConfiguration, mc.lattice.phonons)
                    statistics.acceptedReplicaExchanges += 1
                end
            end
        end

        

        # if mc.sweep>=mc.thermalizationSweeps && mc.sweep%mc.sampleRate==0 && enableSamples
            
        #     writeMonteCarlo(samplesOutfile*"." * string(Int(mc.sweep/mc.sampleRate)), mc)

        # end

     
        mc.energySeries[mc.sweep+1]=energy/length(mc.lattice)


        #increment sweep
        statistics.sweeps += 1
        mc.sweep += 1

        #runtime statistics
        t = time()
        if mc.sweep % mc.reportInterval == 0
            #collect statistics
            progress = 100.0 * mc.sweep / totalSweeps
            thermalized = (mc.sweep >= mc.thermalizationSweeps) ? "YES" : "NO"
            sweeprate = statistics.sweeps / (t - statistics.initializationTime)
            sweeptime = 1.0 / sweeprate
            eta = (totalSweeps - mc.sweep) / sweeprate

            localUpdateAcceptanceRate = 100.0 * statistics.acceptedLocalUpdates / statistics.attemptedLocalUpdates
            if enableMPI
                replicaExchangeAcceptanceRate = 100.0 * statistics.acceptedReplicaExchanges / statistics.attemptedReplicaExchanges
                allLocalAppectanceRate = zeros(commSize)
                allLocalAppectanceRate[rank + 1] = localUpdateAcceptanceRate
                MPI.Allgather!(UBuffer(allLocalAppectanceRate, 1), MPI.COMM_WORLD)
                allReplicaExchangeAcceptanceRate = zeros(commSize)
                allReplicaExchangeAcceptanceRate[rank + 1] = replicaExchangeAcceptanceRate
                MPI.Allgather!(UBuffer(allReplicaExchangeAcceptanceRate, 1), MPI.COMM_WORLD)
            end

            #print statistics
            if rank == 0
                str = ""
                str *= @sprintf("Sweep %d / %d (%.1f%%)", mc.sweep, totalSweeps, progress)
                str *= @sprintf("\t\tETA : %s\n", Dates.format(Dates.now() + Dates.Second(round(Int64,eta)), "dd u yyyy HH:MM:SS"))
                str *= @sprintf("\t\tthermalized : %s\n", thermalized)
                str *= @sprintf("\t\tsweep rate : %.1f sweeps/s\n", sweeprate)
                str *= @sprintf("\t\tsweep duration : %.3f ms\n", sweeptime * 1000)
                
                if enableMPI
                    for n in 1:commSize
                        str *= @sprintf("\t\tsimulation %d update acceptance rate: %.2f%%\n", n - 1, allLocalAppectanceRate[n])
                        str *= @sprintf("\t\tsimulation %d replica exchange acceptance rate : %.2f%%\n", n - 1, allReplicaExchangeAcceptanceRate[n])
                    end
                else
                    str *= @sprintf("\t\tupdate acceptance rate: %.2f%%\n", localUpdateAcceptanceRate)
                end
                str *= @sprintf("\n")
                print(str)
            end

            #reset statistics
            statistics = MonteCarloStatistics()
        end

        #write checkpoint
        if enableOutput
            checkpointPending = time() - lastCheckpointTime >= mc.checkpointInterval
            enableMPI && (checkpointPending = MPIBcastBool(checkpointPending, 0, MPI.COMM_WORLD))
            if checkpointPending
                writeMonteCarlo(outfile, mc)
                lastCheckpointTime = time()
                rank == 0 && @printf("Checkpoint written on %s.\n", Dates.format(Dates.now(), "dd u yyyy HH:MM:SS"))
            end
        end
    end

    #write final checkpoint
    if enableOutput
        writeMonteCarlo(outfile, mc)
        rank == 0 && @printf("Checkpoint written on %s.\n", Dates.format(Dates.now(), "dd u yyyy HH:MM:SS"))
    end
    
    #return
    rank == 0 && @printf("Simulation finished on %s.\n", Dates.format(Dates.now(), "dd u yyyy HH:MM:SS"))
    return nothing    
end