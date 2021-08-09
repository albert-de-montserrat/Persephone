Base.:+(A::Point2D{Cartesian},B::Point2D{Cartesian}) = Point2D{Cartesian}(A.x + B.x , A.z + B.z)

Base.:+(A::Point2D{Polar},B::Point2D{Polar}) = Point2D{Polar}(A.x + B.x , A.z + B.z)

Base.:*(A::Point2D{Polar},b::Number) = Point2D{Polar}(A.x * b , A.z * b) 

Base.:*(A::Point2D{Cartesian},b::Number) = Point2D{Cartesian}(A.x * b , A.z * b) 

# PARTICLES ADVECTION ===============================================================
function particlesadvection!(Particles, Δt; multiplier=1)
    Threads.@threads for i in axes(Particles,1)
        @inbounds Particles[i].CCart.x += Particles[i].UCart.x * Δt * multiplier
        @inbounds Particles[i].CCart.z += Particles[i].UCart.z * Δt * multiplier
    end
end


function particle_coordinates(P::Vector{PINFO}; coordinates_system = "cartesian")
    np = length(P)
    xp = Vector{Float64}(undef, np)
    zp = similar(xp)

    if coordinates_system == "cartesian"
        Threads.@threads for i in 1:np
            @inbounds xp[i] = P[i].CCart.x
            @inbounds zp[i] = P[i].CCart.z
        end

    elseif coordinates_system == "polar"
        Threads.@threads for i in 1:np
            @inbounds xp[i] = P[i].CPolar.x
            @inbounds zp[i] = P[i].CPolar.z
        end

    end

    return xp, zp
end

function particle_coordinates(P::PINFO)
    xp = P.CCart.x
    zp = P.CCart.z
    return xp, zp
end

function particle_coordinates!(xp, zp, P::Vector{PINFO})
    Threads.@threads for i in eachindex(P)
        @inbounds xp[i] = P[i].CCart.x
        @inbounds zp[i] = P[i].CCart.z
    end
end

#= INTERPOLATION FROM INTEGRATION POINT TO PARTICLE + utilities ======================
    EXTRAPOLATION FROM INTEGRATION POINTS TO NODAL POSITIONS
    Local coordinates of triangular element
          (0,1)                   η
            |\                    ^
            | \                   |
            |  \                  |
            |   \                ------> ξ
            |    \                |
            |     \
    (0,1/2) |      \ (1/2,1/2)
            |       \
            |        \
            |         \
            |          \
            |           \
            |____________\
        (0,0) (1/2,0) (1,0)
    New triangular coordinates for triangular elements in ip framework:
        xi'  = 2xi-1/3   --> xi  = xi'/2 + 1/6
        eta' = 2eta-1/3  --> eta = eta'/2 + 1/6
    Value at nodal coordinates
        w_i_vertex = sum(w_i_ip * N_i_node)
=========================================================================================#
function Fij2particle(particle_fields, particle_info, particle_weights, EL2NOD_P1, EL2NOD, F)
    
    # (1) IP to NODE ----------------------------------------------------------
    # -- Nodal element coordinates
    ξ = [0.0 ,1.0, 0.0]
    η = [0.0 ,0.0, 1.0]

    # -- Transformation to ip triangular element coordinates
    ξv = @. 2.0*ξ - 1.0/3.0
    ηv = @. 2.0*η - 1.0/3.0

    # -- Shape functions
    nn3 = Val(3)
    NN, _ = shape_functions_triangles([ξv ηv],nn3)
    
    # -- Map from integration points to the 6 element nodes 
    m, n = size(F)
    Fxx = Array{Float64,2}(undef, m, n)
    Fzz, Fxz, Fzx = similar(Fxx), similar(Fxx), similar(Fxx)
    @inbounds Threads.@threads for i in CartesianIndices(F)
         Fxx[i] = F[i][1,1]
         Fxz[i] = F[i][1,2]
         Fzx[i] = F[i][2,1]
         Fzz[i] = F[i][2,2]
    end

    Fxx_nodal = nodalvalue(Fxx, NN)
    Fzz_nodal = nodalvalue(Fzz, NN)
    Fxz_nodal = nodalvalue(Fxz, NN)
    Fzx_nodal = nodalvalue(Fzx, NN)

    # -- Smooth fields
    nnod = maximum(EL2NOD)
    w = Vector{Float64}(undef,nnod) # weight buffer
    dummy = similar(w) # output buffer
    Fxx_smooth = smoothfield(Fxx_nodal,EL2NOD,w,dummy)
    Fzz_smooth = smoothfield(Fzz_nodal,EL2NOD,w,dummy)
    Fxz_smooth = smoothfield(Fxz_nodal,EL2NOD,w,dummy)
    Fzx_smooth = smoothfield(Fzx_nodal,EL2NOD,w,dummy)

    kernelF2particle!(
        particle_fields,
        Fxx_smooth,
        Fzz_smooth,
        Fxz_smooth,
        Fzx_smooth,
        EL2NOD_P1,
        particle_info,
        particle_weights
        )

    return particle_fields
end

@inline function barycentric_particle(v::NamedTuple, pc::Point2D{T}) where {T}
    x1,x2,x3 = v.x[1], v.x[2], v.x[3]
    z1,z2,z3 = v.z[1], v.z[2], v.z[3]
    x,z = pc.x, pc.z
    b1 = ((z2 - z3) * (x - x3) + (x3 - x2) * (z - z3)) /
         ((z2 - z3) * (x1 - x3) + (x3 - x2) * (z1 - z3))
    b2 = ((z3 - z1) * (x - x3) + (x1 - x3) * (z - z3)) /
         ((z2 - z3) * (x1 - x3) + (x3 - x2) * (z1 - z3))
    b3 = 1 - b1 - b2
    @SVector [b1, b2, b3]
end 

@inline function dot_F2part(F, t, N, i1, i2, idx)
    a = 0.0
    for (i, _i) in enumerate(idx)
       @inbounds a += F[t, _i][i1, i2]*N[i] 
    end
    a
end

function kernel_Fij_particle!(particle_fields, particle_info, ipx, ipz, F, i)
    t = particle_info[i].t_parent
    idx = [4,6,5]
    v = (x=(view(ipx, t, idx)), z=(view(ipz, t, idx))) # Corner ips := [4 6 5]
    pc = particle_info[i].CPolar
    N = barycentric_particle(v, pc)

    particle_fields.Fxx[i] = dot_F2part(F, t, N, 1, 1, idx)
    particle_fields.Fzx[i] = dot_F2part(F, t, N, 2, 1, idx)
    particle_fields.Fxz[i] = dot_F2part(F, t, N, 1, 2, idx)
    particle_fields.Fzz[i] = dot_F2part(F, t, N, 2, 2, idx)
    
end

function F2particle(particle_fields, particle_info, ipx, ipz, F)
    Threads.@threads for ipart in 1:length(particle_info)
        kernel_Fij_particle!(particle_fields, particle_info, ipx, ipz, F, ipart)
    end
    particle_fields
end

function ip2node(EL2NOD, area_el, F)
    # -- Nodal element coordinates
    ξ = Float64.([0,1,0])
    η = Float64.([0,0,1])

    # -- Transformation to ip triangular element coordinates
    ξv = @. 2.0*ξ - 1.0/3.0
    ηv = @. 2.0*η - 1.0/3.0

    # -- Shape functions
    nn3 = Val(3)
    NN,_ = shape_functions_triangles([ξv ηv], nn3)
    # NN = [SVector{3}(NN[i]) for i=1:3]
    
    # -- Map from integration points to the 6 element nodes 
    F_nodal = nodalvalue(F, NN)
  
    # -- Smooth fields
    nnod = maximum(EL2NOD)
    w = Vector{Float64}(undef,nnod) # weight buffer
    dummy = similar(w) # output buffer
    smoothfield(F_nodal, EL2NOD, w, dummy)
    # smoothfield(F_nodal, view(EL2NOD,1:6,:), area_el, w, dummy)
end


function Fijkernel!(FF, idx1, idx2, NN, EL2NOD, EL2NOD_P1, area_el, w, dummy, 
                   particle_info, particle_weights, particle_fields)
                   
    m, n = size(FF)
    F = Array{Float64,2}(undef, m, n)
    @inbounds for i in CartesianIndices(FF)
         F[i] = FF[i][idx1, idx2]
    end

    # F = [FF[i][idx1,idx2] for i in CartesianIndices(FF)]
    F_nodal = nodalvalue(F, NN)
    F_smooth = smoothfield(F_nodal, view(EL2NOD,1:6,:), w, dummy)
    # smoothfield(F_nodal, EL2NOD, area_el, w, dummy)

    @inbounds for i in axes(particle_info,1)
        idx = view(EL2NOD_P1,:,particle_info[i].t)
        particle_fields.Fxx[i] = node2particle(view(F_smooth,idx), particle_weights[i].barycentric)
    end
end

function Fij2particle_thread!(particle_fields,particle_info,particle_weights,EL2NOD_P1,EL2NOD,
    area_el,F)
    
    # (1) IP to NODE ----------------------------------------------------------
    # -- Nodal element coordinates
    ξ = Float64.([0,1,0])
    η = Float64.([0,0,1])

    # -- Transformation to ip triangular element coordinates
    ξv = @. 2.0*ξ - 1.0/3.0
    ηv = @. 2.0*η - 1.0/3.0

    # -- Shape functions# -- Shape functions
    nn3 = Val(3)
    NN,_ = shape_functions_triangles([ξv ηv], nn3)
    
    # -- Map from integration points to the 6 element nodes 
    w  = Vector{Float64}(undef,nnod) # weight buffer
    dummy = similar(w) # output buffer
    
    Fijkernel!(F, 1, 1, NN, EL2NOD, EL2NOD_P1, area_el, w, dummy, particle_info, particle_weights, particle_fields)  
    Fijkernel!(F, 1, 2, NN, EL2NOD, EL2NOD_P1, area_el, w, dummy, particle_info, particle_weights, particle_fields)  
    Fijkernel!(F, 2, 1, NN, EL2NOD, EL2NOD_P1, area_el, w, dummy, particle_info, particle_weights, particle_fields)  
    Fijkernel!(F, 2, 2, NN, EL2NOD, EL2NOD_P1, area_el, w, dummy, particle_info, particle_weights, particle_fields)  

end

"""
Interpolate smoothed nodal Fij to particles
"""
function kernelF2particle!(particle_fields,Fxx_smooth,Fzz_smooth,Fxz_smooth,Fzx_smooth, EL2NOD_P1,particle_info,particle_weights)
    @inbounds Threads.@threads for i in axes(particle_info,1)
        idx = view(EL2NOD_P1,:,particle_info[i].t)
        particle_fields.Fxx[i] = node2particle(view(Fxx_smooth, idx), particle_weights[i].barycentric)
        particle_fields.Fzz[i] = node2particle(view(Fzz_smooth, idx), particle_weights[i].barycentric)
        particle_fields.Fxz[i] = node2particle(view(Fxz_smooth, idx), particle_weights[i].barycentric)
        particle_fields.Fzx[i] = node2particle(view(Fzx_smooth, idx), particle_weights[i].barycentric)
    end
end

# function smoothfield(A,EL2NOD,area_el,w,A_smooth) 
#     fill!(A_smooth,0.0)
#     fill!(w,0.0)
#     @inbounds for i in 1:6
#         idx = view(EL2NOD,i,:)
#         e2ni = idx'
#         dummy = accumarray(e2ni,area_el)
#         @views w[idx] += view(dummy,idx)

#         dummy = accumarray(e2ni,view(A,:,i).*area_el)
#         @views A_smooth[idx] += view(dummy,idx)
#     end
#     A_smooth./w
# end

function smoothfield(F, EL2NOD, w, dummy) 
    fill!(dummy, 0.0)
    fill!(w, 0)
    for i in 1:6
        idx = view(EL2NOD,i,:)
        @views w[idx] .+= 1
        @views dummy[idx] .+= view(F, :, i)
    end
    dummy./w
end

"""
Extrapolate field from integration point to nodal positions
    (*) A field @ integration points
    (*) NN linear shape functions
"""
@inline function nodalvalue(A, NN)
    A3 = takecornerips(A)
    Anodal = Array{Float64,2}(undef, size(A3,1), size(A,2))
    for u in CartesianIndices(A3)
        @inbounds Anodal[u] = mydot(NN[u.I[2]], view(A3, u.I[1],:))
    end
    averageedge!(Anodal)
    return Anodal
end

"""
Take values at the linear integration points 
"""
@inline function takecornerips(A)
    m,n = size(A,1), Int(size(A,2)/2)
    B   = Array{Float64,2}(undef,m,n)
    for i in 1:m
        @inbounds B[i,1] = A[i,4]
        @inbounds B[i,2] = A[i,6]
        @inbounds B[i,3] = A[i,5]
    end
    B
end

takecornerips(A::Array{SArray{Tuple{2,2},Float64,2,4},2}) = @views A[:, [4, 6, 5]]

"""
Average field at nodes located on the edges of the element
"""
@inline function averageedge!(A)
    for i in axes(A,1)
        @inbounds A[i,4] = 0.5 * (A[i,1] + A[i,2])
        @inbounds A[i,5] = 0.5 * (A[i,3] + A[i,2])
        @inbounds A[i,6] = 0.5 * (A[i,1] + A[i,3])
    end
end

"""
node2particle(A::Vector{T}, barycentric::Vector{T})

    Linear interpolation from element nodes to arbitrary point within the element using barycentric coordinates (equivalent linear  FEM shape functions.)

        A := field defined at triangular vertices
        N := barycentric coordinates of i-th particle
"""
@inline node2particle(A_p1,N) = mydot(A_p1,N)

# INTERPOLATE VELOCITIES FROM NODES TO PARTICLES =======================================

function velocities2ip(Ucartesian, EL2NOD)
    # =========== PREPARE INTEGRATION POINTS & DERIVATIVES wrt LOCAL COORDINATES
    nip = 6
    nnodel = 3
    nel = size(EL2NOD,2)
    ni, nn = Val(nip), Val(nnodel)
    N, dNds, _, w_ip  = _get_SF(ni,nn)
    Ux_ip = Array{Float64,2}(undef,nel,nip)
    Uz_ip = similar(Ux_ip)
    
    for iel in 1:nel
        Ux_el, Uz_el = getvelocity(Ucartesian, view(EL2NOD,:,iel))
        @inbounds for ip in 1:nip
            Ux_ip[iel, ip] = mydot(Ux_el[1:3], N[ip])
            Uz_ip[iel, ip] = mydot(Uz_el[1:3], N[ip])
        end

    end
    return Ux_ip, Uz_ip
end

function getvelocity(U,idx)
    Ux = @SVector [U[idx[1]].x, U[idx[2]].x, U[idx[3]].x, U[idx[4]].x, U[idx[5]].x, U[idx[6]].x]
    Uz = @SVector [U[idx[1]].z, U[idx[2]].z, U[idx[3]].z, U[idx[4]].z, U[idx[5]].z, U[idx[6]].z]
    return Ux, Uz
end

getvelocityX(U,idx) = @SVector [U[idx[1]].x, U[idx[2]].x, U[idx[3]].x]
getvelocityZ(U,idx) = @SVector [U[idx[1]].z, U[idx[2]].z, U[idx[3]].z]

function getvelocity(U)
    Ux = @inbounds [u.x for u in U]
    Uz = @inbounds [u.z for u in U]
    return Ux,Uz
end

# INTERPOLATE TEMPERATURE FROM NODES TO PARTICLES =======================================
@inline function initial_particle_temperature!(particle_fields, e2n_p1, T, particle_info, particle_weights)
    np = length(particle_info)    
    # loop
    Threads.@threads for i in 1:np
        # -- Temperature interpolation (weighted average)
        @inbounds particle_fields.T[i] = weightednode2particle(
            view(T, view(e2n_p1,:,particle_info[i].t)),
            particle_weights[i].barycentric
            )
    end

end

function interpolate_temperature!(
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
                    
    @timeit to "T → particle" begin
        @timeit to "step 1" particle_fields = temperature2particle(
            T0,
            particle_fields,
            gr.e2n_p1,
            ρ,
            T,
            particle_info,
            particle_weights,
            VarT,
            nθ,
            nr,
            Δt,
        )
        @timeit to "step 2" ΔT_remaining = ΔTsubgrid2node(
            ΔT,
            particle_fields,
            particle_info,
            particle_weights,
            gr
        )
        @timeit to "step 3" corrected_particle_temperature!(
            particle_fields, ΔT_remaining
        )
    end
end

# INTERPOLATE TEMPERATURE FROM NODES TO PARTICLES =======================================
@inline function temperature2particle(T0, particle_fields, e2n_p1, ρ, T, particle_info, particle_weights, VarT, nθ, nr, Δt)

    Cp, κ = VarT.Cp, VarT.κ

    # allocations
    nt = Threads.nthreads()
    Tᵢ0 = Vector{Float64}(undef, nt)
    # Tᵢ = Vector{Float64}(undef, nt)
    # ρᵢ = Vector{Float64}(undef, nt)

    # constant variables
    Δθ = 2π*1.22/(nθ)
    Δr = 1/(nr-1)
    #=
        Gerya's book, pp. 143
        Δt_diff = Cp * ρₘ / (κₘ * (2/Δx² + Δz²) )
        0 ≤ d ≤ 1 , numerical diffusion coefficient
    =#
    multiplier = Δt/(Cp/(κ*(2/((Δθ/2)^2) + 2/((Δr/2)^2)))) # == Δt/Δt_diff
    d = 1

    # loop
    Threads.@threads for i in eachindex(particle_info)

        # -- Store old T
        @inbounds Tᵢ0[Threads.threadid()] = weightednode2particle(
            view(T0, view(e2n_p1,:,particle_info[i].t)),
            particle_weights[i].barycentric
            )
        
        # Δt_diff = ρᵢ[Threads.threadid()]*multiplier
        @inbounds particle_fields.ΔT_subgrid[i] = (Tᵢ0[Threads.threadid()] - particle_fields.T[i])*(1 - exp(-d*multiplier))
        @inbounds particle_fields.T[i] += particle_fields.ΔT_subgrid[i]

    end

    return particle_fields

end

function corrected_particle_temperature!(particle_fields, ΔT_remaining)
    Threads.@threads for i in eachindex(ΔT_remaining)
        @inbounds particle_fields.T[i] += ΔT_remaining[i]
    end
end

function F2ip(F, particle_fields, particle_info, particle_weights, nel)
    np = length(particle_info)
    weight = Array{Float64,2}(undef, np, 6)
    @inbounds Threads.@threads for i in axes(particle_info,1)
        ## THIS IS TYPE UNSTABLE -> FIX ASAP
        @views weight[i,:] .= 1.0./particle_weights[i].ip_weights # inverse of the distance
    end

    F2ip_inner_tkernel(F, particle_info, particle_fields, weight, nel)

    # # weight times field
    # t = Vector{Int32}(undef, np)
    # wxx, wzz, wxz, wzx =  similar(weight), similar(weight), similar(weight), similar(weight)
    # Threads.@threads for i in axes(weight,1)
    #     for j in axes(weight,2)
    #         @inbounds w = weight[i,j]
    #         @inbounds wxx[i,j] = w*particle_fields.Fxx[i]
    #         @inbounds wzz[i,j] = w*particle_fields.Fzz[i]
    #         @inbounds wxz[i,j] = w*particle_fields.Fxz[i]
    #         @inbounds wzx[i,j] = w*particle_fields.Fzx[i]
    #     end
    #    @inbounds t[i] = Int32(particle_info[i].t_parent) # t-th element hosting particle i-ith 
    # end
    
    # # Interpolate from particle to integration point
    # for j in 1:6
    #     accw = 1.0./accumarray(t,view(weight,:,j)) # weights sum
    #     Fxx  = accumarray(t,view(wxx,:,j)).*accw
    #     Fzz  = accumarray(t,view(wzz,:,j)).*accw
    #     Fxz  = accumarray(t,view(wxz,:,j)).*accw
    #     Fzx  = accumarray(t,view(wzx,:,j)).*accw
    #     Threads.@threads for i in 1:nel
    #         @inbounds F[i,j] = @SMatrix [Fxx[i] Fxz[i]; Fzx[i] Fzz[i]]
    #     end
    # end

    return F
end

function F2ip_inner_kernel(F, particle_info, particle_fields, weight, nel)
    Fxx = fill(0.0, nel, 6)
    Fzz = fill(0.0, nel, 6)
    Fzx = fill(0.0, nel, 6)
    Fxz = fill(0.0, nel, 6)
    for i in axes(weight,1)
        ω = ntuple(j->weight[i,j]::Float64, 6)
        @inbounds @views @. Fxx[particle_info[i].t_parent, :] += particle_fields.Fxx[i]* ω
        @inbounds @views @. Fxz[particle_info[i].t_parent, :] += particle_fields.Fxz[i]* ω
        @inbounds @views @. Fzx[particle_info[i].t_parent, :] += particle_fields.Fzx[i]* ω
        @inbounds @views @. Fzz[particle_info[i].t_parent, :] += particle_fields.Fzz[i]* ω
    end
    Threads.@threads for i in eachindex(Fxx)
        @inbounds F[i] = @SMatrix [Fxx[i] Fxz[i]; Fzx[i] Fzz[i]]
    end
    return F
end

function F2ip_inner_tkernel(F, particle_info, particle_fields, weight, nel)
    nt = Threads.nthreads()
    Fxx = [fill(0.0, nel, 6) for _ in 1:nt]
    Fzz = [fill(0.0, nel, 6) for _ in 1:nt]
    Fzx = [fill(0.0, nel, 6) for _ in 1:nt]
    Fxz = [fill(0.0, nel, 6) for _ in 1:nt]
    accw = [fill(0.0, nel, 6) for _ in 1:nt]

    # Acummarray on each thread
    Threads.@threads for i in axes(weight,1)
        ω = ntuple(j->weight[i,j], 6)
        @inbounds @views @. Fxx[Threads.threadid()][particle_info[i].t_parent, :] += particle_fields.Fxx[i]* ω
        @inbounds @views @. Fxz[Threads.threadid()][particle_info[i].t_parent, :] += particle_fields.Fxz[i]* ω
        @inbounds @views @. Fzx[Threads.threadid()][particle_info[i].t_parent, :] += particle_fields.Fzx[i]* ω
        @inbounds @views @. Fzz[Threads.threadid()][particle_info[i].t_parent, :] += particle_fields.Fzz[i]* ω
        @inbounds @views @. accw[Threads.threadid()][particle_info[i].t_parent, :] += ω
    end

    # Acummulate on first array
    for i in 2:nt
        @inbounds Fxz[1] .+= Fxz[i]
        @inbounds Fxx[1] .+= Fxx[i]
        @inbounds Fzx[1] .+= Fzx[i]
        @inbounds Fzz[1] .+= Fzz[i]
        @inbounds accw[1] .+= accw[i]
    end

    # Store F tensor
    Threads.@threads for i in CartesianIndices(Fxx[1])
        @inbounds F[i] = @SMatrix [Fxx[1][i]/accw[1][i] Fxz[1][i]/accw[1][i]; Fzx[1][i]/accw[1][i] Fzz[1][i]/accw[1][i]]
    end
    return F
end

function F2ip_inner_tkernel_unsafe(F, particle_info, particle_fields, weight, nel)
    Fxx = fill(0.0, nel, 6)
    Fzz = fill(0.0, nel, 6)
    Fzx = fill(0.0, nel, 6)
    Fxz = fill(0.0, nel, 6)
    accw = fill(0.0, nel, 6)
    # Acummarray on each thread
    Threads.@threads for i in axes(weight,1)
        ω = ntuple(j->weight[i,j], 6)
        @inbounds @views @. Fxx[particle_info[i].t_parent, :] += particle_fields.Fxx[i]* ω
        @inbounds @views @. Fxz[particle_info[i].t_parent, :] += particle_fields.Fxz[i]* ω
        @inbounds @views @. Fzx[particle_info[i].t_parent, :] += particle_fields.Fzx[i]* ω
        @inbounds @views @. Fzz[particle_info[i].t_parent, :] += particle_fields.Fzz[i]* ω
        @inbounds @views @. accw[particle_info[i].t_parent, :] += ω
    end

    # Store F tensor
    Threads.@threads for i in CartesianIndices(Fxx)
        @inbounds F[i] = @SMatrix [Fxx[i]/accw[i] Fxz[i]/accw[i]; Fzx[i]/accw[i] Fzz[i]/accw[i]]
    end
    return F
end

#=
    Interpolate from particles to nodes:
        T_node = ∑T_particle_i*w_i / ∑w_i
    where
        w = 1/distance(node -> ip)^n
=#
function T2node(T, particle_fields, particle_info, particle_weights, gr, IDs)
    
    EL2NOD_P1 = gr.e2n_p1

    #=  Build arrays  =#
    t = [particle_info[i].t for i in axes(particle_info,1)]
    w = Array{Float64,2}(undef, size(particle_weights,1),size(particle_weights[1].node_weights,1))
    for i in axes(particle_weights,1) 
        dummy = SVector{3}(particle_weights[i].barycentric)
        for j in 1:3
            # @inbounds w[i,j] = 1/(dummy[j]^2)
            @inbounds w[i,j] = dummy[j]
        end
    end

    # T2node_inner_tkernel(T, particle_info, particle_fields, w, EL2NOD_P1)
    # applybounds!(T, 1.0, 0.0) 
    # T

    nodes = transpose(view(EL2NOD_P1,:,t))
    A = w.*particle_fields.T
    nn = Array(nodes)
    A1 = kernel2(nn, A, true)
    A2 = kernel2(nn, w, true)
    T = vec(A1./A2)
    fixT!(T, 0.0, 1.0, IDs)
    applybounds!(T, 1.0, 0.0) 
    T
end

function T2node_inner_tkernel_unsafe(T, particle_info, particle_fields, weight, EL2NOD_P1)
    fill!(T, 0.0)
    accw = fill(0.0, length(T))
    # Acummarray on each thread
    Threads.@threads for i in axes(weight,1)
        ω = ntuple(j->weight[i,j], 3)
        @inbounds @views T[view(EL2NOD_P1,:,particle_info[i].t)] .+= particle_fields.T[i].*ω
        @inbounds @views accw[view(EL2NOD_P1,:,particle_info[i].t)] .+= ω
    end

    # Store T tensor
    Threads.@threads for i in eachindex(T)
        @inbounds T[i] /= accw[i]
    end
    return T
end

function T2node_inner_tkernel(T, particle_info, particle_fields, weight, EL2NOD_P1)
    fill!(T, 0.0)
    nt = Threads.nthreads()
    Tnt = [fill(0.0, length(T)) for _ in 1:nt-1]
    accw = [fill(0.0, length(T)) for _ in 1:nt]
    # Acummarray on each thread
    Threads.@threads for i in axes(weight,1)
        ω = ntuple(j->weight[i,j], 3)
        if Threads.threadid() == 1
            @inbounds @views T[view(EL2NOD_P1,:,particle_info[i].t)] .+= particle_fields.T[i].*ω
        else
            @inbounds @views Tnt[Threads.threadid()-1][view(EL2NOD_P1,:,particle_info[i].t)] .+= particle_fields.T[i].*ω
        end
        @inbounds @views accw[Threads.threadid()][view(EL2NOD_P1,:,particle_info[i].t)] .+= ω
    end

    # Acummulate on first array
    for i in 2:nt
        @inbounds T .+= Tnt[i-1]
        @inbounds accw[1] .+= accw[i]
    end

    # Store T tensor
    Threads.@threads for i in eachindex(T)
        @inbounds T[i] /= accw[1][i]
    end
    return T
end

function ΔTsubgrid2node(ΔT, particle_fields, particle_info, particle_weights, gr)

    EL2NOD_P1, EL2NOD = gr.e2n_p1, gr.e2n
    	
    #=  Build arrays  =#
    t = [p.t for p in particle_info]
    w = Matrix{Float64}(undef, size(particle_weights,1), size(particle_weights[1].node_weights,1))
    # t = Vector{Int64}(undef, size(particle_weights,1))
    A = similar(w)
    Threads.@threads for i in axes(particle_weights,1) 
        for j in 1:3
            @inbounds ω = particle_weights[i].barycentric[j]
            @inbounds w[i,j] = ω
            @inbounds A[i,j] = ω*particle_fields.ΔT_subgrid[i]
        end
    end
    
    # nodes = transpose(view(EL2NOD_P1,:,t))    
    # nn = Array(nodes)
    nn = view(EL2NOD_P1, :, t)'
    A1 = kernel2(nn, A, true) # accumarray -> ∑Tᵢ*ωᵢ
    A2 = kernel2(nn, w, true) # accumarray -> ∑ωᵢ
    ΔT_subgrid = vec(A1./A2) # -> ∑Tᵢ*ωᵢ/∑ωᵢ
    ΔT_remaining = ΔT .- ΔT_subgrid
    
#    ΔT_remaining = similar(A1)
#    @inbounds for i in eachindex(A1)
#	ΔT_remaining[i] = ΔT[i] - A1[i][1]/A2[i][1]
#    end
    
    nodalfield2partcle(ΔT_remaining, EL2NOD_P1, particle_weights, particle_info) # interpolate to particles
 
    # # ΔT_subgrid in element vertices node
    # ΔT_subgrid_vert = vec(A1./A2) # -> ∑Tᵢ*ωᵢ/∑ωᵢ
    # # Add ΔT_subgrid in center node
    # ΔT_subgrid_el = view(ΔT_subgrid_vert, view(EL2NOD,1:6,:)) # element T
    # ΔT_subgrid_bubble = vec(mean(ΔT_subgrid_el, dims = 1))
    # ΔT_subgrid = vcat(ΔT_subgrid_vert, ΔT_subgrid_bubble)

end

"""
nodalfield2partcle(field, e2n, particle_weights, particle_info)

    Interpolate field from node to corresponding particle
"""
function nodalfield2partcle(field, e2n, particle_weights, particle_info)
    # memory allocations
    np = length(particle_info)
    field_particle = Vector{Float64}(undef, np)

    # interpolate from node to corresponding particle
    Threads.@threads for i in 1:np
        @inbounds field_particle[i] = weightednode2particle(
            (@views field[e2n[:, particle_info[i].t]]),
            particle_weights[i].barycentric
        )
    end
    field_particle
end


"""
    applybounds!(A::AbstractArray{T}, upper::T, lower::T) where {T<:Real}
"""
function applybounds!(A::AbstractArray{T}, upper::T, lower::T) where {T<:Real}
    Threads.@threads for i in eachindex(A)
        @inbounds A[i] = ifelse(A[i] > upper, upper, A[i])
        @inbounds A[i] = ifelse(A[i] < lower, lower, A[i])
    end
end

"""
    applybounds(A::T, upper::T,lower::T) where {T<:Real}
"""
applybounds(A::T, upper::T, lower::T) where {T<:Real} = max(min(A, upper), lower)

ip2particle(ip_field,iw) = mydot(ip_field,iw) / sum(iw)

function kernel1(a, b, collapse=false)
    m,n = maximum(a),size(a,2) 
    v = fill(0.0,m,n)
    for j in axes(v,2), _ in axes(b,1)
        @inbounds v[a[i,j],j] += b[i][j]
    end
    collapse == true ? sum(v,dims=2) : v
end

function kernel2(a, b, collapse=false)
    m, n = maximum(a), size(a,2) 
    v = fill(0.0,m,n)
    Threads.@threads for j in axes(v,2)
        for i in axes(b,1)
            @inbounds v[a[i,j],j] += b[i,j]
        end
    end
    collapse == true ? sum(v,dims=2) : v
end

@inline function T2particle(particle_fields, e2n_p1, T, particle_info, particle_weights)
    @inbounds Threads.@threads for i in axes(particle_info,1) 
        # -- Temperature interpolation (weighted average)
        particle_fields.T[i] = weightednode2particle(view(T, view(e2n_p1,:,particle_info[i].t)),
                                                     fw.(particle_weights[i].barycentric))
    end
   return particle_fields
end

@inline fw(x) = x^3 * exp(-3*x)

# @inline weightednode2particle(A,ω) = (ω[1]*A[1] + ω[2]*A[2] + ω[3]*A[3]) / (ω[1]+ω[2]+ω[3])

@inline function weightednode2particle(A, ω)
    a, b = 0.0, 0.0
    @turbo for i in axes(A,1)
        a += ω[i]*A[i]
        b += ω[i]
    end
    a/b
end

function T2particle_cubic!(particle_fields, gr, T, particle_info)
    Tp = quasicubic_interpolation(gr, particle_info, T)
    Threads.@threads for i in axes(Tp,1)
        @inbounds particle_fields.T[i] = max(min(Tp[i], 1.0), 0.0)
    end
end
