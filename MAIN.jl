# using GLMakie
import Pkg; Pkg.activate(".")
using Persephone
using LinearAlgebra, TimerOutputs
import Statistics:mean

function main()

    #=========================================================================
        INITIALISE OUTPUT PATHS
    =========================================================================#
    iscluster = false
    if iscluster
        path = "/storage2/unipd/navarro/AnnulusBenchmarks"
    else
        path = "/home/albert/Desktop/output"
    end
    folder = "NoHealingTrue"
    OUT, iplot = setup_output(path, folder)

    #=========================================================================
        RELOAD CHECKPOINT    
    =========================================================================#
    load = false
    iload = 131
    i0 = 1

    #=========================================================================
        MAKE GRID
    =========================================================================#
    gr, IDs = init_grid(4; split=2) # 1st argument = N
    PhaseID = 1
    min_inradius = inrectangle(gr)/2
    GlobC = [Point2D{Polar}(gr.θ[i], gr.r[i]) for i in 1:gr.nnod] # → global coordinates

    #=========================================================================
        Boundary conditions
    =========================================================================#
    # T and U boundary conditions
    TBC, UBC = init_BCs(gr, IDs, Ttop = 0.0, Tbot = 1.0, type="free slip" )

    #=========================================================================
        Delete existing statistics file    
    =========================================================================#
    ScratchNu, stats_file  = setup_metrics(gr, path, folder, load = load)

    #=========================================================================
        GET DEM STRUCTURE:    
    =========================================================================#
    dem_file = joinpath("DEM", "DEM_1e-3_vol30.h5")
    # dem_file = joinpath("newDEM", "Dem_1e-3_vol30.h5")
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
    isotropic_idx = findall(ipz .> transition)
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
    # Allocate velocity and pressure fields
    U, P = init_U_P(gr)
    # Allocate velocity gradient, stress and strain tensors:
    F, _, τ, ε,  = initstress(gr.nel)
    # FSE = [FiniteStrainEllipsoid(1.0, 0.0, 0.0, 1.0, 1.0, 1.0) for _ in 1:gr.nel, _ in 1:6]
    # Allocate viscosity tensor:
    𝓒 = 𝓒init(gr.nel, 7)
    # Allocate nodal velocities
    Ucartesian, Upolar = initvelocity(gr.nnod)
    # Initialise temperature @ nodes
    perturbation = :random
    T = init_temperature(gr, IDs, type = perturbation)
    ΔT = similar(T)
    # Initialise temperature @ particles
    init_particle_temperature!(particle_fields, particle_info, type = perturbation)

    # Annealing rate
    annealing = 1e-3
    
    # Finite Strain structure
    FSE = FiniteStrain(gr.nel,
        nip = 6, 
        ϵ = 0, # a1/a2 at which fabric is destroyed
        annealing_rate = 0, # annealing rate
        r_iso = 0 # depth above which Ω is isotropic
    )

    #= Viscosity type. Options:
        (*) "IsoviscousIsotropic"
        (*) "TemperatureDependantIsotropic"
        (*) "IsoviscousAnisotropic"
        (*) "TemperatureDependantAnisotropic"
    =#
    viscosity_type = :TemperatureDependantAnisotropic

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
    g = 1e6
    𝓒 = anisotropic_tensor(FSE.fse, D, Valη)

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
        gr, gr.nnod, 2, 3, 6, gr.nel, 7
    )

    SF_Stress = stress_shape_functions()

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
        i0 = iload + 1
        load_file = string("file_", iload, ".h5")
        toreload = joinpath(pwd(), path, folder, load_file)
        T, F = reloader(toreload)
        FSE, F = getFSE(F, FSE)
        𝓒 = anisotropic_tensor(FSE.fse, D, Valη)
    end

    #=========================================================================
        START SOLVER
    =========================================================================#
    to = TimerOutput()
    Time = 0.0
    T0 = deepcopy(T)

    for iplot in i0:350
        for _ in 1:75
            reset_timer!(to)

            #= Update material properties =#
            state_equation!(ρ, VarT.α, T)
            getviscosity!(η, T)

            #= Stokes solver using preconditioned-CG =#    
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
                solver = :pardiso
            );

            println("min:max Uθ ", extrema(@views U[1:2:end]))
            println("mean speed ", mean(@views @. (√(U[1:2:end]^2 + U[2:2:end]^2))))
            
            #= Adaptive time-step =#
            Δt = calculate_Δt(Ucartesian, gr.nθ, min_inradius)
            println("Δt = ", Δt)

            #= Stress-Strain postprocessor =#
            @timeit to "F" stress!(F, ε, Ucart, gr.nel, DoF_U, coordinates, SF_Stress, Δt)
            # εII = secondinvariant(ε)

            # isotropic_lithosphere!(F, isotropic_idx)
            # F = healing(F, FSE)
            # @timeit to "FSE" FSE = getFSE(F, FSE)
            # F0, FSE0 = deepcopy(F), deepcopy(FS)
            # @timeit to "FSE" FSE, F = getFSE_healing(F, FSE, ϵ=1e3)
            # @timeit to "FSE" getFSE_annealing!(F, FSE, annealing*(Time+Δt))

            # Finite Strain Ellipsoid calculation
            if FSE.isotropic_domain != 0 # check whether any region of the Earth is always isotropic
                isotropic_lithosphere!(F, isotropic_idx) # force isotropy         
            end
            FSE, F = getFSE(F, FSE)

            #= Compute the viscous tensor =#
            @timeit to "viscous tensor" 𝓒 = anisotropic_tensor(FSE.fse, D, Valη)

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
                @sync begin
                    
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
                            gr.nθ,
                            gr.nr,
                            Δt,
                            ΔT,
                        )
                    end

                    
                end

                @timeit to "advection" particle_info, particle_weights, to = advection_RK2(
                        particle_info,
                        gr,
                        particle_weights,
                        Ucartesian,
                        Δt,
                        θThermal,
                        rThermal,
                        coordinates,
                        IntC,
                        to,
                    )

                println("Min-max particle temperature = ", extrema(particle_fields.T))
            end

            @timeit to "Locate articles" begin
                #= Particle locations =#
                particle_info, particle_weights, found = tsearch_parallel(
                    particle_info, particle_weights, θThermal, rThermal, coordinates, gr, IntC
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
                @timeit to "F → ip" F = F2ip(
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

            # healing!(F, FSE)

            println("mean T after advection  ", mean(T))

            write_stats(U, T, length(particle_info), gr, Time, ScratchNu, stats_file)

            Time += Δt
            show(to; compact=true)
        end

        #= Save output file =#
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
            FSE.fse,
            gr.nθ, 
            gr.nr,
            particle_fields,
            particle_info,
            Time,
            Val(η),
        )
        println(" ...done!")

    end
end

main()