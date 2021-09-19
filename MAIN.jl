import Pkg; Pkg.activate(".")
using Persephone
using LinearAlgebra, TimerOutputs
import Statistics:mean

function main()

    #=========================================================================
        INITIALISE OUTPUT PATHS:    
    =========================================================================#
    iscluster = false
    if iscluster
        path = "/storage2/unipd/navarro/AnnulusBenchmarks"
    else
        path = "/home/albert/Desktop/output"
    end
    folder = "HarmAnisoV30Ra1e5"
    OUT, iplot = setup_output(path, folder)

    #=========================================================================
        MAKE GRID
    =========================================================================#
    split = 2
    N = 3
    if split == 1
        nr = Int(1 + 2^N)
        nθ = Int(12 * 2^N)
        gr = Grid_split1(nθ, nr)
    else
        nr = Int(1+2^N)
        nθ = Int(12*2^N)
        nr = Int(1 + 32)
        nθ = Int(256)
        gr = Grid(nθ, nr)
    end
    IDs = point_ids(gr)
    load = false

    PhaseID = 1
    min_inradius = inradius(gr)

    nU = maximum(gr.e2n)
    nnod = length(gr.θ)
    P = fill(0.0, maximum(gr.e2nP))
    U = fill(0.0, nU .* 2)

    GlobC = [Point2D{Polar}(gr.θ[i], gr.r[i]) for i in 1:gr.nnod] # → global coordinates

    #=========================================================================
        Boundary conditions
    =========================================================================#
    TBC = temperature_bcs(gr, IDs; Ttop = 0.0, Tbot = 1.0)
    UBC = velocity_bcs(gr, IDs; type="free slip")
    
    #=========================================================================
        Delete existing statistics file    
    =========================================================================#
    ScratchNu, stats_file  = setup_metrics(gr, path, folder)

    #=========================================================================
        GET DEM STRUCTURE:    
    =========================================================================#
    dem_file = joinpath("DEM", "DEM_1e-3_vol20_new3.h5")
    dem_file = joinpath("newDEM", "Dem_1e-3_vol30.h5")
    Δη, ϕ = 1e-3, 0.3
    D = getDEM(dem_file, Δη, ϕ)

    #=========================================================================
        FIX θ OF ELEMENTS CROSSING π and reshape also r (i.e. periodic boundaries)
    =========================================================================#
    θStokes, rStokes = elementcoordinate(GlobC, @views(gr.e2n[1:3, :]))
    θThermal, rThermal = elementcoordinate(GlobC, gr.e2n_p1)
    fixangles!(θStokes)
    fixangles!(θThermal)
    coordinates = ElementCoordinates(θStokes, rStokes)

    #=========================================================================
        IPs polar coordinates (Some stuff needs to be deprecated)
    =========================================================================#
    θ6, r6 =  elementcoordinate(GlobC, @views(gr.e2n[1:6, :]))
    θ3 = θStokes 
    fixangles6!(θ6)
    fixangles!(θ3)
    ipx, ipz = getips(gr.e2n, θ6, r6 )
    ix, iz = polar2cartesian(ipx, ipz)
    transition = 2.22-0.2276
    # isotropic_idx = findall(ipz .> transition)
    IntC = [@inbounds(Point2D{Polar}(ipx[i], ipz[i])) for i in CartesianIndices(ipx)] # → ip coordinates

    #=========================================================================
        INITIALISE PARTICLES
    =========================================================================#
    particle_info, particle_weights, particle_fields = particles_generator(
        θThermal, rThermal, IntC, gr.e2n_p1, number_of_particles = 8
    )

    #=========================================================================
        ALLOCATE/INITIALISE FIELDS    
    =========================================================================#
    # Allocate velocity gradient, stress and strain tensors:
    F, _, τ, ε,  = initstress(gr.nel)
    FSE = [FiniteStrainEllipsoid(1.0, 0.0, 0.0, 1.0, 1.0, 1.0) for _ in 1:gr.nel, _ in 1:6]
    # Allocate viscosity tensor:
    𝓒 = 𝓒init(gr.nel, 7)
    # Allocate nodal velocities
    Ucartesian, Upolar = initvelocity(gr.nnod)
    # Initialise temperature @ nodes
    perturbation = :harmonic
    T = init_temperature(gr, IDs, type = perturbation)
    ΔT = similar(T)
    # Initialise temperature @ particles
    init_particle_temperature!(particle_fields, particle_info, type = perturbation)

    viscosity_type = :IsoviscousAnisotropic
    #= Options:
        (*) "IsoviscousIsotropic"
        (*) "TemperatureDependantIsotropic"
        (*) "IsoviscousAnisotropic"
        (*) "TemperatureDependantAnisotropic"
    =#

    # Physical parameters for thermal diffusion
    VarT = thermal_parameters(
        κ = 1.0,
        α = 1e-6,
        Cp = 1.0,
        dQdT = 0.0,
    )

    ρ = state_equation(VarT.α, T)
    η = getviscosity(T, viscosity_type, η = 1.81) 
    	# η = 1 for isotropic
    	# η = 1.81 for anisotropic with phi = 30%
    	# η = 1/0.6899025321942348 for anisotropic with phi = 20%
    Valη = Val(η)
    g = 1e5
    𝓒 = anisotropic_tensor(FSE, D, Valη)

    #=========================================================================
        SOLVER INVARIANTS (FOR AN IMMUTABLE MESH):
            TODO pre-compute the Jacobians
    =========================================================================#
    # Allocate rotation tensor 
    RotationMatrices = rotation_matrix(gr.θ)
    # Allocate sparsity patterns of Stokes block matrices 
    KKidx, GGidx, MMidx, = sparsitystokes(gr)
    # Allocate spasity pattern of thermal diffusion stiffness matrix
    CMidx, _ = sparsitythermal(gr.e2n, 6)

    # Stokes immutables
    KS, GS, MS, FS, DoF_U, DoF_P, nn, SF_Stokes, ScratchStokes = stokes_immutables(
        gr, nnod, 2, 3, 6, gr.nel, 7
    )

    # Diffusion_immutables
    KT, MT, FT, DoF_T, valA, SF_Diffusion, ScratchDifussion = diffusion_immutables(
        gr, 2, 3, 6, 7
    )

    # Color elements
    _, color_list = color_mesh(gr.e2n)

    #=========================================================================
        LOAD PREVIOUS MODELS
    =========================================================================#
    if load == true
        toreload = joinpath(pwd(), path, folder, "file_14.h5")
        T, F = reloader(toreload)
        FSE = getFSE(F, FSE)
        𝓒 = anisotropic_tensor(FSE, D, Valη, ipx)
    end

    #=========================================================================
        START SOLVER
    =========================================================================#
    to = TimerOutput()
    Time = 0.0
    T0 = deepcopy(T)

    for iplot in 1:50
        for _ in 1:50
            reset_timer!(to)

            #= Update material properties =#
            state_equation!(ρ, VarT.α, T)
            getviscosity!(η, T)

            #=
                Stokes solver using preconditioned-CG 
            =#       
            Ucartesian, Upolar, U, Ucart, P, to = solveStokes(
                U,
                P,
                gr,
                Ucartesian,
                Upolar,
                g,
                T,
                η,
                𝓒,
                coordinates,
                RotationMatrices,
                PhaseID,
                UBC,
                KKidx,
                GGidx,
                MMidx,
                to,
            );

            println("min:max Uθ", extrema(@views U[1:2:end]))
            println("mean speed  ", mean(@views @. (√(U[1:2:end]^2 + U[2:2:end]^2))))

            Δt = calculate_Δt(Ucartesian, nθ, min_inradius) # adaptive time-step

            #=
                Stress-Strain postprocessor
            =#
            # F, τ, ε, = stress(
            #     Ucart, T, F, 𝓒, τ, ε, gr.e2n, θStokes, rStokes, η, PhaseID, Δt
            # )
            stress!(F, Ucart, gr.nel, DoF_U, coordinates, 6, SF_Stokes, Δt)
            
            # isotropic_lithosphere!(F, isotropic_idx)
            FSE = getFSE(F, FSE)

            #= Compute the viscous tensor =#
            𝓒 = anisotropic_tensor(FSE, D, Valη)

            #=
                Diffusion solver
            =#
            T, T0, ΔT, to = solveDiffusion_threaded(
                color_list,
                CMidx,
                KT,
                MT,
                FT,
                DoF_T,
                coordinates,
                VarT,
                ScratchDifussion,
                SF_Diffusion,
                valA,
                ρ,
                Δt,
                T,
                T0,
                TBC,
                to,
            )

            @timeit to "Particle advenction" begin
                #=
                Particle advection and mappings
                    Fij : ip   |-> particle
                    T   : node |-> particle
                    Ui  : node |-> particle + advection
                =#

                # @timeit to "F → particle"  particle_fields = 
                #        Fij2particle(particle_fields,
                #                     particle_info,
                #                     particle_weights,
                #                     gr,
                #                     F)

                @timeit to "F → particle" particle_fields = F2particle(
                    particle_fields, particle_info, ipx, ipz, F
                )

                @timeit to "T → particle" begin
                    interpolate_temperature!(
                        T0,
                        particle_fields,
                        gr,
                        ρ,
                        T,
                        particle_info,
                        particle_weights,
                        VarT,
                        nθ,
                        nr,
                        Δt,
                        ΔT,
                    )
                end

                @timeit to "advection" particle_info, to = advection_RK2(
                    particle_info,
                    gr,
                    particle_weights,
                    Ucartesian,
                    Δt,
                    θThermal,
                    rThermal,
                    IntC,
                    to,
                )

                println("Min-max particle temperature = ", extrema(particle_fields.T))
            end

            @timeit to "Locate articles" begin
                #= Particle locations =#
                particle_info, particle_weights, found = tsearch_parallel(
                    particle_info, particle_weights, θThermal, rThermal, gr.neighbours, IntC
                )

                lost_particles = length(particle_info) - sum(found)
                check_corruption!(found, particle_fields)
                println("Lost particles: ", lost_particles, " Corrupted particles: ",  length(particle_info) - sum(found) -lost_particles)

                if lost_particles > 0
                    particle_info, particle_weights, particle_fields = purgeparticles(
                        particle_info, particle_weights, particle_fields, found
                    )
                end

            end

            @timeit to "Particle to node/ip" begin
                #=
                    Map back to original immutable locations
                        Fij : particle -> ip
                        T   : particle -> node
                =#
                F = F2ip(
                    F, particle_fields, particle_info, particle_weights, gr.nel
                )

                T = T2node(
                    T,
                    particle_fields,
                    particle_info,
                    particle_weights,
                    gr,
                    IDs,
                )

                println("\n Min-max nodal temperature = ", extrema(T))
            end

            @timeit to "Add/reject particles" begin
                particle_info, particle_weights, particle_fields = addreject(
                    T,
                    F,
                    gr,
                    θThermal,
                    rThermal,
                    IntC,
                    particle_info,
                    particle_weights,
                    particle_fields,
                    min_num_particles = 3
                )
            end

            println("mean T after advection  ", mean(T))

            write_stats(U, T, length(particle_info), gr, Time, ScratchNu, stats_file)

            Time += Δt
            show(to; compact=true)
        end

        #=
            Save output file
        =#
        println("\n time = ", Time)
        println("\n Saving output...")
        OUT = IOs(path, folder, "file", iplot)
        savedata(
            OUT,
            Upolar,
            Ucartesian,
            T,
            η,
            𝓒,
            ρ,
            F,
            FSE,
            nθ, 
            nr,
            particle_fields,
            particle_info,
            Time,
            Val(η),
        )
        println(" ...done!")

    end
end

main()
