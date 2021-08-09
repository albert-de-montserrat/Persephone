include("Mixer.jl")

function main()

    #=========================================================================
        INITIALISE OUTPUT PATHS:    
    =========================================================================#
    iscluster = false
    if iscluster
        path = "/storage2/unipd/navarro/AnnulusBenchmarks"
        folder = "Anisotropic_1e4_2"
    else
        path = "output"
        folder = "test"
    end
    filename = "file"
    iplot = Int32(0)
    OUT = IOs(path, folder, filename, iplot)
    mkpath(joinpath(path, folder))

    #=========================================================================
        MAKE GRID
    =========================================================================#
    split = 2
    N = 4
    if split == 1
        nr = Int(1 + 2^N)
        nθ = Int(12 * 2^N)
        gr = Grid_split1(nθ, nr)
    else
        nr = Int(1+2^N)
        nθ = Int(12*2^N)
        # nr = Int(1 + 32)
        # nθ = Int(256)
        gr = Grid(nθ, nr)
    end
    IDs = point_ids(gr)
    load = false

    PhaseID = 1
    min_inradius = inradius(gr)

    GlobC = [Point2D{Polar}(gr.θ[i], gr.r[i]) for i in 1:gr.nnod] # → global coordinates

    #=========================================================================
        Boundary conditions
    =========================================================================#
    TBC = temperature_bcs(gr, IDs; Ttop = 0.0, Tbot = 1.0)
    UBC = velocity_bcs(gr, IDs; type="free slip")
    
    #=========================================================================
        SETUP BLAS THREADS:    
    =========================================================================#
    # LinearAlgebra.BLAS.set_num_threads(1) 

    #=========================================================================
        Delete existing statistics file    
    =========================================================================#
    ScratchNu = ScratchNusselt(gr)
    stats_file = joinpath(pwd(), path, folder, "statistics.txt")
    if isfile(stats_file)
        rm(stats_file)
    end

    #=========================================================================
        GET DEM STRUCTURE:    
    =========================================================================#
    fname = joinpath("DEM", "DEM_1e-3_vol20_new3.h5")
    D = getDEM(fname)

    # =========================================================================
    nU = maximum(gr.e2n)
    nnod = length(gr.θ)
    P = fill(0.0, maximum(gr.e2nP))
    U = fill(0.0, nU .* 2)

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
    θ6, =  elementcoordinate(GlobC, @views(gr.e2n[1:6, :]))
    θ3 = θStokes 
    fixangles6!(θ6)
    fixangles!(θ3)
    ipx, ipz = getips(gr.e2n, θStokes, rStokes)
    IntC = [@inbounds(Point2D{Polar}(ipx[i], ipz[i])) for i in CartesianIndices(ipx)] # → ip coordinates

    #=========================================================================
        INITIALISE PARTICLES
    =========================================================================#
    particle_info, particle_weights = particles_generator(
        θThermal, rThermal, IntC, gr.e2n_p1
    )
    particle_fields = init_pvars(length(particle_info))

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
    # Initialise temperature
    IDs = domain_ids(GlobC)
    T = init_temperature(gr, IDs)
    init_particle_temperature!(particle_fields, particle_info)
    ΔT = similar(T)

    viscosity_type = "TemperatureDependantIsotropic"
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
    # η = getviscosity(T, viscosity_type; η=1.81) η=1 is the default
    η = getviscosity(T, viscosity_type, η = 1)
    Valη = Val(η)
    g = 1e5
    𝓒 = anisotropic_tensor(FSE, D, Valη, ipx)

    #=========================================================================
        SOLVER INVARIANTS (FOR AN IMMUTABLE MESH):
            TODO pre-compute the Jacobians
    =========================================================================#
    # Allocate rotation tensor 
    TT = rotation_matrix(gr.θ)
    # Allocate sparsity patterns of Stokes block matrices 
    KKidx, GGidx, MMidx, = sparsitystokes(gr)
    # Allocate spasity pattern of thermal diffusion stiffness matrix
    CMidx, _ = sparsitythermal(gr.e2n, 6)

    # # Stokes immutables
    # KS, GS, MS, FS, DoF_U, DoF_P, nn, SF_Stokes, ScratchStokes = stokes_immutables(
    #     gr, nnod, 2, 3, 6, gr.nel, 7
    # )

    # Diffusion_immutables
    KT, MT, FT, DoF_T, valA, SF_Diffusion, ScratchDifussion = diffusion_immutables(
        gr, 2, 3, 6, 7
    )
    #(gr.e2n, nnod, ndim, nvert, nnodel, gr.nel, nip)
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

    for iplot in 1:250
        for _ in 1:50
            reset_timer!(to)

            #= Update material properties =#
            @timeit to "Material properties" begin
                state_equation!(ρ, VarT.α, T)
                getviscosity!(η, T)
            end

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
                TT,
                PhaseID,
                UBC,
                KKidx,
                GGidx,
                MMidx,
                to,
            )

            println("min:max Uθ", extrema(@views U[1:2:end]))
            println("mean speed  ", mean(@views @. (√(U[1:2:end]^2 + U[2:2:end]^2))))

            Δt = calculate_Δt(Ucartesian, nθ, min_inradius) # adaptive time-step

            @timeit to "Stress" begin
                #=
                    Stress-Strain postprocessor
                =#
                F, τ, ε, τII, εII = stress(
                    Ucart, T, F, 𝓒, τ, ε, gr.e2n, θStokes, rStokes, η, PhaseID, Δt
                )
                # shear_heating = shearheating(τ, ε)
            end

            @timeit to "Finite Strain Ellipsoid" begin
                FSE = getFSE(F, FSE)
                # isotropic_lithosphere!(FSE, ipz)
            end

            #= Compute the viscous tensor =#
            @timeit to "Get and rotate viscous tensor" begin
                𝓒 = anisotropic_tensor(FSE, D, Valη, ipx)
            end

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

                @timeit to "F → particle"  particle_fields = 
                       Fij2particle(particle_fields,
                                    particle_info,
                                    particle_weights,
                                    gr.e2n_p1,
                                    gr.e2n,
                                    F)

                # @timeit to "F → particle" particle_fields = F2particle(
                #     particle_fields, particle_info, ipx, ipz, F
                # )
    
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
                        to
                    )
                end

                @timeit to "advection" particle_info, to = advection_RK2(
                    particle_info,
                    gr.e2n_p1,
                    particle_weights,
                    Ucartesian,
                    Δt,
                    θThermal,
                    rThermal,
                    gr.neighbours,
                    IntC,
                    to,
                )

                println("Min-max particle temperature = ", extrema(particle_fields.T))
            end

            @timeit to "Particles" begin
                #= Particle locations =#
                @timeit to "Locate" particle_info, particle_weights, found = tsearch_parallel(
                    particle_info, particle_weights, θThermal, rThermal, gr.neighbours, IntC
                )

                lost_particles = length(particle_info) - sum(found)
                println("Lost particles: ", lost_particles)

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
                @timeit to "Fij -> ip" F = F2ip(
                    F, particle_fields, particle_info, particle_weights, gr.nel
                )

                @timeit to "T -> node" T = T2node(
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
                )
            end

            println("mean T after advection  ", mean(T))

            @timeit to "Run stats" write_stats(U, T, gr, Time, ScratchNu, stats_file)

            Time += Δt
            show(to; compact=true)
        end

        #=
            Save output file
        =#
        println("\n time = ", Time)
        println("\n Saving output...")
        OUT = IOs(path, folder, filename, iplot)
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
            gr.e2n,
            GlobC,
            particle_fields,
            particle_info,
            Time,
            Val(η),
        )
        println(" ...done!")
    end
end

main()
