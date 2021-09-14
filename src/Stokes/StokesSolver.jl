struct RotationMatrix{T}
    TT::T
    TTᵀ::T
end

# ==================================================================================================================
function solveStokes(U, P, gr, Ucartesian,Upolar, g, ρ, η, 𝓒,
    coordinates, RotationMatrices,
    PhaseID, UBC,
    KKidx,GGidx,MMidx,
    to)

    ∂Ωu = UBC.Ω
    ufix = UBC.vfix
    ifree = UBC.ifree
    
    @timeit to "Stokes" begin
        @timeit to "Assembly" KK,GG,MM,Rhs =
           assembly_stokes_cylindric(gr, coordinates, g, ρ, η, 𝓒, PhaseID, KKidx, GGidx, MMidx)

        @timeit to "BCs" begin
            KK,GG,Rhs = _prepare_matrices(KK, GG, Rhs, RotationMatrices)
            U,Rhs = _apply_bcs(U,KK,Rhs,∂Ωu,ufix)
        end

        @timeit to "PCG solver" U, P = StokesPcCG(U, P, KK, MM, GG, Rhs, ifree)

        @timeit to "Remove net rotation" U, Ucart, Upolar, Ucartesian = 
            updatevelocity2(U, Ucartesian, Upolar, ρ, RotationMatrices.TT, coordinates, gr)
    end

    return Ucartesian, Upolar, U, Ucart, P, to
end
# ==================================================================================================================

@inline initvelocity(nn) = [Point2D{Cartesian}(0.0,0.0) for _ = 1:nn], [Point2D{Polar}(0.0,0.0) for _ = 1:nn] 

function remove_wind(r, Uθ::Vector{Point2D{Polar}}) 
    rmax = maximum(r)
    id = findall(x->x==rmax,r)
    mean([@inbounds Uθ[i].x for i in id])
end

function cartesianvelocity!(U, Ucart, θ, r)
    for (c, i) in enumerate(1:2:(length(U)-1))
        Ucart[i] = U[i+1]*sin(θ[c]) + r[c]*U[i]*cos(θ[c])
        Ucart[i+1] = U[i+1]*cos(θ[c]) - r[c]*U[i]*sin(θ[c])
    end
end

@inline function updatevelocity(U, Ucartesian, Upolar, θ, r, TT)
    nU = length(U)
    Ucart = similar(U)
    # -- Fill tangential and radial velocities
    @inbounds for (c,i) in enumerate(1:2:nU-1)        
        Upolar[c].x = U[i]
        Upolar[c].z = U[i+1]
    end

    # -- Remove net rotation
    U_wind = remove_wind(r, Upolar)
    invr = @. (r-1.22)
    @views (@. U[1:2:nU-1] -= U_wind*invr)
    extrema(@. U[1:2:nU-1] - U_wind)
    # a, b = extrema(@views U[1:2:nU-1])
    # c = 0.5*(a+b)
    # @views (@. U[1:2:nU-1] -= c*invr)

    # -- Convert to Cartesian
    Ucart = TT*U
    
    # -- Fill cartesian velocities
    @inbounds for (c, i) in enumerate(1:2:nU-1)
        Upolar[c].x = U[i]
        # Ucart[i] = Upolar[c].z*sin(θ[c]) + r[c]*Upolar[c].x*cos(θ[c])
        # Ucart[i+1] = Upolar[c].z*cos(θ[c]) - r[c]*Upolar[c].x*sin(θ[c])
        Ucartesian[c].x = Ucart[i]
        Ucartesian[c].z = Ucart[i+1]
    end
    return U, Ucart, Upolar, Ucartesian
end

modulus(v::Vector{T}) where T = √(dot(v, v))

modulus(v::NTuple{N, T}) where {N,T} = √(dot(v, v))

modulus(v::NTuple{2, T}) where {T} = √(v[1]*v[1] + v[2]*v[2])

function modulus(v::Vector{T}, u::Vector{T}) where T
    m = similar(v)
    Threads.@threads for i in eachindex(v)
        @inbounds m[i] = modulus(
            (u[i], v[i])
        )
    end
    m
end

function cross_product(v::Vector{T}, u::Vector{T}) where T
    m = similar(v)
    Threads.@threads for i in eachindex(v)
        @inbounds m[i] = cross_product(
            (v[i], u[i]),
            (v[i], u[i]),
        )
    end
    m
end

"""
    2D cross-product : ||u||⋅||v||⋅sin(θ) 
"""
function cross_product(x, v)
    mx, mv = modulus(x), modulus(v) # vector modulus
    θ = acos(dot(x, v)/(mx*mv)) # angle between vectors
    mx*mv*sin(θ) # 2D cross product
end

@inline angular_velocity(x, v) = modulus(v)*sin(x[1])/x[2] # 2D cross product

function v_perp(gr, Ux, Uz)
    v = similar(Ux)
    for i in eachindex(Ux)
        v[i] = cross_product(
            (gr.x[i], gr.z[i]),
            (Ux[i], Uz[i]),
        )
    end
end

function v_perp(gr, U::Vector{Point2D{T}}) where {T}
    v = Vector{Float64}(undef, length(U))
    Threads.@threads for i in eachindex(U)
        @inbounds v[i] = angular_velocity(
            (gr.θ[i], gr.r[i]),
            (U[i].x, U[i].z),
        )
    end
    v
end

function update_velocity_structures(U, Upolar, Ucartesian, TT)
    nU = length(U)
    # Cartesian velocity
    Ucart = TT*U
    Threads.@threads for i in 1:2:nU-1
        @inbounds Upolar[div(i,2)+1].x = U[i]
        @inbounds Upolar[div(i,2)+1].z = U[i+1]
        @inbounds Ucartesian[div(i,2)+1].x = Ucart[i]
        @inbounds Ucartesian[div(i,2)+1].z = Ucart[i+1]
    end
    return Upolar, Ucartesian, Ucart
end

## Nullspace removal
@inline function updatevelocity2(U, Ucartesian, Upolar, ρ, TT, coordinates, gr)

    θ, r = coordinates.θ, coordinates.r

    # Net rotation
    Uθ = @views U[1:2:end]
    angular_momentum = volume_integral(Uθ.*ρ, gr.e2n, θ, r)
    moment = volume_integral(gr.r.*ρ,  gr.e2n, θ, r)
    ω = angular_momentum/moment
    @views @. U[1:2:end] -= ω*gr.r

    # Update velocity
    Upolar, Ucartesian, Ucart = update_velocity_structures(U, Upolar, Ucartesian, TT)

    return U, Ucart, Upolar, Ucartesian
end
 
@inline function _apply_bcs(U,KK,Rhs,∂Ω,vfix)
    U[∂Ω] = vfix # write prescribed temperature values
    Rhs -= KK[:,∂Ω] * vfix
    return U,Rhs
end

@inline function _prepare_matrices(KK,GG,Fb,RotationMatrices)
    KK = SparseMatrixCSC(Symmetric(KK,:L))
    dropzeros!(KK)
    dropzeros!(GG)   
    KK = RotationMatrices.TTᵀ*KK*RotationMatrices.TT
    GG = RotationMatrices.TTᵀ*GG
    Fb = RotationMatrices.TTᵀ*Fb
    KK = SparseMatrixCSC(Symmetric(KK,:L))
    return KK,GG,Fb
end

@inline function rotation_matrix(theta)
    npt   = length(theta)
    # -- memory ALLOCATIONS
    nnz_T = 4 * npt
    KKi_T = Vector{Int64}(undef,nnz_T) 
    KKj_T = similar(KKi_T)
    KK1_T = fill(1.0,nnz_T)

    # Diagonal slots: cos(theta). This will use up the first 2*npt slots
    # KKi_T[1:2*npt]      = 1:2*npt
    # KKj_T[1:2*npt]      = 1:2*npt
    @inbounds for i in 1:2*npt
        KKi_T[i] = i
        KKj_T[i] = i
    end

    idx     = 1:2:2*npt-1
    idx2    = 3*npt+1:4*npt

    @inbounds @fastmath for (ii, i) in enumerate(2*npt+1:3*npt)
        cost           = cos(theta[ii]) 
        sint           = sin(theta[ii]) 
        i0             = idx[ii]
        i2             = idx2[ii]
        # Diagonal slots: cos(theta). This will use up the first 2*npt slots
        KK1_T[i0]      = cost   # element (1,1) in the rotation matrix [TT]
        KK1_T[i0+1]    = cost   # element (2,2) in the rotation matrix [TT]
        # Off-diagonal slots: sin(theta) and -sin(theta). This will use up the last 2*npt slots
        KKi_T[i]       = i0
        KKj_T[i]       = i0+1
        KK1_T[i]       = sint  # element (1,2) in the rotation matrix [TT] (ORIGINAL: flipped sign -> transposed rotation)
        # KK1_T[i]       = -sint  # element (1,2) in the rotation matrix [TT] 
        KKi_T[i2]      = i0+1
        KKj_T[i2]      = i0
        KK1_T[i2]      = -sint # element (2,1) in the rotation matrix [TT] (ORIGINAL:flipped sign -> transposed rotation)
        # KK1_T[i2]      = sint # element (2,1) in the rotation matrix [TT] 
    end
    
    TT = sparse(KKi_T,KKj_T,KK1_T)
    transTT = SparseMatrixCSC(TT')

    return RotationMatrix(TT, transTT)

end

function assembly_stokes_cylindric(gr, coordinates, g, ρ, η, 𝓒, PhaseID,KKidx,GGidx,MMidx) 
    
    EL2NOD, EL2NODP, theta, r = gr.e2n, gr.e2n_p1, coordinates.θ, coordinates.r
    
    # ============================================ MODEL AND BLOCKING PARAMETERS
    ndim            = 2
    # nvert           = 3
    nnodel          = size(EL2NOD,1)
    nel             = size(EL2NOD,2)
    nelblk          = min(nel, 1200)
    nblk            = ceil(Int,nel/nelblk)
    il              = one(nelblk); iu = nelblk
    # =========== PREPARE INTEGRATION POINTS & DERIVATIVES wrt LOCAL COORDINATES
    nip             = 7
    # nUdof           = ndim*maximum(EL2NOD)
    # nPdof           = maximum(EL2NODP)
    nUdofel         = ndim * nnodel
    nPdofel         = size(EL2NODP,1)
    ni, nn, nnP, nn3 = Val(nip), Val(nnodel), Val(3), Val(3)
    N,dNds,~,w_ip   = _get_SF(ni,nn)
    NP,~,~,~        = _get_SF(ni,nnP)
    N3,~,dN3ds,_    = _get_SF(ni,nn3)
    # ============================================================= ALLOCATIONS
    detJ_PL         = Vector{Float64}(undef,nelblk)
    R_31            = similar(detJ_PL) # =  detJa_PL*dxi_dth
    R_21            = similar(detJ_PL) # = -detJa_PL*deta_dth
    Th_31           = similar(detJ_PL) # = -detJa_PL*dxi_dr
    Th_21           = similar(detJ_PL) # =  detJa_PL*deta_dr
    # NxN             = Matrix{Float64}(undef,3,3)
    # ang_dist        = Matrix{Float64}(undef,3,nelblk)
    # VCOORD_th       = similar(ang_dist)
    # VCOORD_r        = similar(ang_dist)
    dNdx            = Matrix{Float64}(undef,nelblk,size(dNds[1],2))
    dNdy            = similar(dNdx)
    ω               = Vector{Float64}(undef,Int(nelblk))
    invJx_double    = Matrix{Float64}(undef,Int(nelblk),ndim)    # storage for x-components of Jacobi matrix
    invJz_double    = similar(invJx_double)                     # storage for z-components of Jacobi matrix  
    # ==================================== STORAGE FOR DATA OF ELEMENTS IN BLOCK
    K_blk           = fill(0.0, nelblk, Int(nUdofel*(nUdofel+1)/2))
        # symmetric stiffness matrix, dim=vel dofs, but only upper triangle
    M_blk           = fill(0.0, nelblk, Int(nPdofel*(nPdofel+1)/2))
        # symmetric mass matrix, dim=pressure nodes, but only upper triangle
    G_blk           = fill(0.0, nelblk, nPdofel*nUdofel)
        # asymmetric gradient matrix, vel dofs x pressure nodes
    Fb_blk          = fill(0.0, nelblk, nUdofel)
        # storage for entries in bouyancy force vector    

    # ====================================  STORAGE FOR DATA OF ALL ELEMENT MATRICES/VECTORS
    K_all           = Matrix{Float64}(undef, Int(nUdofel*(nUdofel+1)/2),nel)
    M_all           = Matrix{Float64}(undef, Int(nPdofel*(nPdofel+1)/2),nel)
    G_all           = Matrix{Float64}(undef, nUdofel*nPdofel,nel)
    Fb_all          = Matrix{Float64}(undef, nUdofel,nel)

    #=========================================================================
    BLOCK LOOP - MATRIX COMPUTATION
    =========================================================================#
    @inbounds for ib in 1:nblk
        #===========================================================================
        CALCULATE JACOBIAN, ITS DETERMINANT AND INVERSE

        NOTE: For triangular elements with non-curved edges the Jacobian is
              the same for all integration points (i.e. calculated once
              before the integration loop). Further, linear 3-node shape are
              sufficient to calculate the Jacobian.
        ===========================================================================#
        VCOORD_th = view(theta,:,il:iu)
        VCOORD_r = view(r,:,il:iu)

        J_th = gemmt(VCOORD_th', dN3ds')
        J_r = gemmt(VCOORD_r', dN3ds')

        _fill_R_J!(J_th,J_r,VCOORD_th,VCOORD_r,
            R_31,R_21,Th_31,Th_21,detJ_PL)

        # ------------------------ NUMERICAL INTEGRATION LOOP (GAUSS QUADRATURE)        
        _ip_loop!(K_blk, M_blk, G_blk, Fb_blk,
            g, η, 𝓒, ρ, EL2NOD, PhaseID,il:iu,
            dNds, w_ip, nip, nnodel, nelblk, nPdofel,nUdofel,
            dNdx,dNdy,N,N3,NP,
            ω,invJx_double, invJz_double, detJ_PL,
            R_21,R_31,Th_21,Th_31,
            VCOORD_th,VCOORD_r)

        # --------------------------------------- WRITE DATA INTO GLOBAL STORAGE
        @views begin
            K_all[:,il:iu] .= K_blk'
            G_all[:,il:iu] .= G_blk'
            M_all[:,il:iu] .= M_blk'
            Fb_all[:,il:iu] .= Fb_blk'
        end
        
        # -------------------------------------------------- RESET BLOCK MATRICES
        fill!(K_blk, 0.0)
        fill!(G_blk, 0.0)
        fill!(M_blk, 0.0)
        fill!(Fb_blk, 0.0)
        
        # -------------------------------- READJUST START, END AND SIZE OF BLOCK
        il += nelblk
        if ib == nblk-1
           # ------------ Account for different number of elements in last block
           nelblk       = nel - il + 1  # new block size
           # --------------------------------- Reallocate at blocks at the edge 
           detJ_PL      = Vector{Float64}(undef,nelblk)
           R_31         = similar(detJ_PL) 
           R_21         = similar(detJ_PL) 
           Th_31        = similar(detJ_PL) 
           Th_21        = similar(detJ_PL) 
           dNdx         = Matrix{Float64}(undef,nelblk,size(dNds[1],2))
           dNdy         = similar(dNdx)
           K_blk        = fill(0.0, nelblk, Int(nUdofel*(nUdofel+1)/2))
           M_blk        = fill(0.0, nelblk, Int(nPdofel*(nPdofel+1)/2))
           G_blk        = fill(0.0, nelblk, nPdofel*nUdofel)                
           Fb_blk       = fill(0.0, nelblk, nUdofel)
           ω            = Vector{Float64}(undef,Int(nelblk))
           invJx_double = Matrix{Float64}(undef,Int(nelblk),ndim) 
           invJz_double = similar(invJx_double) 
        end
        iu += nelblk

    end # end block loop
    
    # #===========================================================================
    # ASSEMBLY OF GLOBAL SPARSE MATRICES AND RHS-VECTOR
    # ===========================================================================#
    KK, GG, MM, Fb = _create_stokes_matrices(EL2NOD,nUdofel,ndim,
        KKidx, GGidx, MMidx,
        K_all, G_all, M_all, Fb_all)

    return KK,GG,MM,Fb
    
end # END OF ASSEMBLY FUNCTION

#===============================================================================
                        BEGINING OF ASSEMBLYE SUBFUNCTIONS
===============================================================================#
@inline function _create_stokes_matrices(EL2NOD,EL2NODP,nUdofel,nPdofel,ndim,
    K_all,G_all,M_all,Fb_all)

    nel     = size(EL2NOD,2)
    indx_j  = Array{Int64}(undef,nUdofel,nUdofel)
    indx_i  = Array{Int64}(undef,nUdofel,nUdofel)
    dummy   = 1:nUdofel
    @inbounds for i in 1:nUdofel
        indx_j[i,:] = dummy
        indx_i[:,i] = dummy'
    end

    # indx_i      = indx_j';
    indx_i      = tril(indx_i); indxx_i = vec(indx_i); filter!(x->x>0,indxx_i)
    indx_j      = tril(indx_j); indxx_j = vec(indx_j); filter!(x->x>0,indxx_j)

    #==========================================================================
    ASSEMBLY OF GLOBAL STIFFNESS MATRIX
    ==========================================================================#
    EL2DOF                    = fill(0,nUdofel, nel)
    EL2DOF[1:ndim:nUdofel,:] .= @. ndim*(EL2NOD-1)+1
    EL2DOF[2:ndim:nUdofel,:] .= @. ndim*(EL2NOD-1)+2
    K_i        = deepcopy(EL2DOF[vec(indxx_i),:])
    K_j        = deepcopy(EL2DOF[vec(indxx_j),:])
    indx       = K_i .< K_j
    tmp        = deepcopy(K_j[indx])
    K_j[indx]  = deepcopy(K_i[indx])
    K_i[indx]  = tmp
    #  convert triplet data to sparse global stiffness matrix (assembly)
    KK         = sparse(vec(K_i) , vec(K_j), vec(K_all))

    #==========================================================================
    ASSEMBLY OF GLOBAL GRADIENT MATRIX
    ==========================================================================#
    G_i    = repeat(EL2DOF,nPdofel,1) # global velocity dofs
    indx_j = repeat(collect(1:nPdofel)',nUdofel,1)
    G_j    = EL2NODP[indx_j,:]        # global pressure nodes
    # convert triplet data to sparse global gradient matrix (assembly)
    GG     = sparse(vec(G_i) , vec(G_j) ,vec(G_all))

    #==========================================================================
    ASSEMBLY OF GLOBAL FORCE VECTOR
    ==========================================================================#
    Fb     = accumarray(vec(EL2DOF),vec(Fb_all));

    #==========================================================================
    ASSEMBLY OF GLOBAL (1/VISCOSITY)-SCALED MASS MATRIX
    ==========================================================================#
    indx_j  = Array{Int64}(undef,nPdofel,nPdofel)
    indx_i  = Array{Int64}(undef,nPdofel,nPdofel)
    dummy   = collect(1:nPdofel)
    @inbounds for i in 1:nPdofel
        indx_j[i,:] = dummy
        indx_i[:,i] = dummy'
    end
    indx_i    = tril(indx_i); indxx_i = vec(indx_i); filter!(x->x>0,indxx_i)
    indx_j    = tril(indx_j); indxx_j = vec(indx_j); filter!(x->x>0,indxx_j)
    M_i       = deepcopy(EL2NODP[indxx_i,:]); MM_i = vec(M_i);
    M_j       = deepcopy(EL2NODP[indxx_j,:]); MM_j = vec(M_j);
    indx      = MM_i .< MM_j
    tmp       = deepcopy(MM_j[indx])
    M_j[indx] = deepcopy(MM_i[indx])
    M_i[indx] = tmp;
    # convert triplet data to sparse global matrix
    MM        = sparse(vec(M_i) , vec(M_j) , vec(M_all))

    return KK,GG,MM,Fb

end

@inline function _create_stokes_matrices(EL2NOD,nUdofel,ndim,
    KKidx,GGidx,MMidx,
    K_all,G_all,M_all,Fb_all)

    nel     = size(EL2NOD,2)
  
    #==========================================================================
    ASSEMBLY OF GLOBAL STIFFNESS MATRIX
    ==========================================================================#
    #  convert triplet data to sparse global stiffness matrix (assembly)
    KK     = sparse(KKidx._i ,KKidx._j, vec(K_all))

    #==========================================================================
    ASSEMBLY OF GLOBAL GRADIENT MATRIX
    ==========================================================================#
    # convert triplet data to sparse global gradient matrix (assembly)
    GG     = sparse(GGidx._i ,GGidx._j ,vec(G_all))

    #==========================================================================
    ASSEMBLY OF GLOBAL FORCE VECTOR
    ==========================================================================#
    EL2DOF = Array{Int64}(undef,nUdofel, nel)
    @views EL2DOF[1:ndim:nUdofel,:] .= @. ndim*(EL2NOD-1)+1
    @views EL2DOF[2:ndim:nUdofel,:] .= @. ndim*(EL2NOD-1)+2
    Fb     = accumarray(vec(EL2DOF),vec(Fb_all));

    #==========================================================================
    ASSEMBLY OF GLOBAL (1/VISCOSITY)-SCALED MASS MATRIX
    ==========================================================================#
    # convert triplet data to sparse global matrix
    MM     = sparse(MMidx._i ,MMidx._j , vec(M_all))

    return KK,GG,MM,Fb

end

@inline function _fill_R_J!(J_th,J_r,VCOORD_th,VCOORD_r,
    R_31,R_21,Th_31,Th_21,detJ_PL)

    VCOORD_r1 = view(VCOORD_r,1,:)
    VCOORD_r2 = view(VCOORD_r,2,:)
    VCOORD_r3 = view(VCOORD_r,3,:)
    VCOORD_th1 = view(VCOORD_th,1,:)
    VCOORD_th2 = view(VCOORD_th,2,:)
    VCOORD_th3 = view(VCOORD_th,3,:)

    @turbo for i in 1:length(R_31)
        detJ_PL[i] = J_th[i,1]*J_r[i,2] - J_th[i,2]*J_r[i,1]
        R_31[i]    = VCOORD_r3[i]  - VCOORD_r1[i]  # =  detJa_PL*dxi_dth
        R_21[i]    = VCOORD_r2[i]  - VCOORD_r1[i]  # = -detJa_PL*deta_dth
        Th_31[i]   = VCOORD_th3[i] - VCOORD_th1[i] # = -detJa_PL*dxi_dr
        Th_21[i]   = VCOORD_th2[i] - VCOORD_th1[i] # =  detJa_PL*deta_dr
    end

    return R_31,R_21,Th_31,Th_21,detJ_PL
end

@inline function _ip_loop!(K_blk, M_blk, G_blk, Fb_blk,
    g, η, 𝓒, ρ, EL2NOD, PhaseID,els,
    dNds, w_ip, nip, nnodel, nelblk, nPdofel,nUdofel,
    dNdx,dNdy,N,N3,NP,
    ω,invJx_double, invJz_double, detJ_PL,
    R_21,R_31,Th_21,Th_31,
    VCOORD_th,VCOORD_r)

    sin_ip  = similar(detJ_PL)
    cos_ip  = similar(detJ_PL)
    th_ip = similar(detJ_PL)
    r_ip = similar(detJ_PL)
    NP_blk = Matrix{Float64}(undef, length(detJ_PL), 3)

    @inbounds for ip=1:nip

        N_ip = N3[ip]
        # NP_blk = repeat(NP[ip],nelblk,1)
        NP_blk .= NP[ip].*ones(nelblk)

        #=======================================================================
        PROPERTIES OF ELEMENTS AT ip-TH EVALUATION POINT
        =======================================================================#
        Dens_blk    = _element_density(ρ, EL2NOD, PhaseID, els, NP[ip])
        Visc_blk    = _element_viscosity(η, EL2NOD, PhaseID, els, NP[ip])
        #  Gravitational force at ip-th integration point
        Fg_blk      = g * Dens_blk

        #===========================================================================================================
        CALCULATE 2nd JACOBIAN (FROM CARTESIAN TO POLAR COORDINATES --> curved edges), ITS DETERMINANT AND INVERSE
        
        NOTE: For triangular elements with curved edges the Jacobian needs to be computed at each integration
               point (inside the integration loop).
        ===========================================================================================================#

        gemmt!(th_ip,  VCOORD_th', N_ip')
        gemmt!(r_ip,  VCOORD_r', N_ip')

        _derivative_weights!(dNds[ip],ω,dNdx,dNdy,w_ip[ip],th_ip,r_ip,sin_ip,cos_ip,
            R_21,R_31,Th_21,Th_31,detJ_PL,invJx_double,invJz_double)
        
        _fill_Kblk!(K_blk, 𝓒, Visc_blk, Val(η),
            ω, dNdx, dNdy, nnodel, els, ip)

        _fill_Gblk!(G_blk,NP_blk,
            ω, dNdx, dNdy, nPdofel,nUdofel,nnodel)

        _fill_Mblk!(M_blk,Visc_blk,NP_blk,
            ω, nPdofel)

        _fill_Fbblk!(Fb_blk,Fg_blk, N[ip],
            ω, sin_ip,cos_ip,nUdofel)

    end # end integration point loop
end # ENF OF IP_LOOP FUNCTION


@inline function _derivative_weights!(dNds,ω,dNdx,dNdy,w_ip,th_ip,r_ip,sin_ip,cos_ip,
    R_21,R_31,Th_21,Th_31,detJ_PL,invJx_double,invJz_double)

    # --- Start loop (faster and more memory friendly than single - line declarations)
    for i in 1:length(th_ip)
        # -------------------------------- i-th extractions for some extra speed
        detJ      = detJ_PL[i]
        invdetJ   = 1/detJ
        sin_ip0   = sin(th_ip[i])
        cos_ip0   = cos(th_ip[i])
        sin_ip[i] = sin_ip0
        cos_ip[i] = cos_ip0
        R_31_i    = R_31[i]
        R_21_i    = R_21[i]
        Th_31_i   = Th_31[i]
        Th_21_i   = Th_21[i]
        r_i       = r_ip[i]
        invr_i    = 1/r_i
        # --------------------------------------- Inverse of the double Jacobian
        invJx_double[i,1] = ( R_31_i*cos_ip0 - Th_31_i*sin_ip0*r_i )* invr_i * invdetJ
        invJx_double[i,2] = (-R_21_i*cos_ip0 + Th_21_i*sin_ip0*r_i )* invr_i * invdetJ
        invJz_double[i,1] = (-R_31_i*sin_ip0 - Th_31_i*cos_ip0*r_i )* invr_i * invdetJ
        invJz_double[i,2] = ( R_21_i*sin_ip0 + Th_21_i*cos_ip0*r_i )* invr_i * invdetJ
        # ---------------------------- Numerical integration of element matrices
        ω[i]  = r_i * detJ * w_ip
    end

    # --------------------------------------- Derivatives wrt global coordinates
    mul!(dNdx, invJx_double, dNds)
    mul!(dNdy, invJz_double, dNds)

end

@inline function _fill_Kblk!(K_blk, 𝓒, Visc_blk, ::Val{Isotropic}, ω, dNdx, dNdy, nnodel,els,ip) 
    C1 = 4/3; # Used instead of 'D' matrix in standard assembly
    C2 = 2/3; # see Zienkiewicz book, Vol 2, 4th edition, p 519

    # C1 = 2; # incompressible
    # C2 = 0; # 
    indx = 1
    nn = size(K_blk,1)
    for i in 1:nnodel 
        # |-> ugly as fuck, but faster and more memory friendly than MATLAB style semivectorised original version
        for j in i:nnodel            
            # x-velocity equation (1st, 3rd, 5th,... rows of stiffness matrices)            
            @inbounds @simd for k in 1:nn
                # x-velocity (1st, 3th, 5th,... columns)
                K_blk[k,indx] +=  Visc_blk[k]*ω[k] *
                    ( C1 * dNdx[k,i]*dNdx[k,j] + dNdy[k,i]*dNdy[k,j])                                                
            end            
            indx   += 1
            @inbounds @simd for k in 1:nn
                # y-velocity equation (2nd, 4th, 6th,... rows of stiffness matrices)
                K_blk[k,indx] +=  Visc_blk[k]*ω[k] *
                    ( -C2 * dNdx[k,i]*dNdy[k,j] + dNdy[k,i]*dNdx[k,j])                                
            end
            indx   += 1
        end

        for j in i:nnodel
            if j > i                
                # x-velocity equation (3rd, 5th, 7th... rows of stiffness matrices)            
                @inbounds @simd for k in 1:nn
                    # x-velocity (1st, 3th, 5th,... columns)
                    K_blk[k,indx] +=  Visc_blk[k]*ω[k] * 
                        (-C2 * dNdy[k,i]*dNdx[k,j] + dNdx[k,i]*dNdy[k,j])                                
                end
                indx   += 1
            end
            @inbounds @simd for k in 1:nn
                # y-velocity equation (2nd, 4th, 6th,... rows of stiffness matrices)
                K_blk[k,indx] +=   Visc_blk[k]*ω[k] *
                    ( C1 * dNdy[k,i]*dNdy[k,j] + dNdx[k,i]*dNdx[k,j]) 
            end
            indx += 1
        end
        
    end

end

@inline function _fill_Kblk!(K_blk, 𝓒, Visc_blk, ::Val{Anisotropic}, ω, dNdx, dNdy, nnodel,els,ip) 

    # Unpack viscous tensor
    η11 = view(𝓒.η11,els,ip).*Visc_blk
    η22 = view(𝓒.η33,els,ip).*Visc_blk
    η33 = view(𝓒.η55,els,ip).*Visc_blk
    η12 = view(𝓒.η13,els,ip).*Visc_blk
    η13 = view(𝓒.η15,els,ip).*Visc_blk.*1
    η23 = view(𝓒.η35,els,ip).*Visc_blk.*1

    # C = 0.5; # Used instead of 'D' matrix in standard assembly
    C = 1 # Used instead of 'D' matrix in standard assembly
    indx = 1
    nn = size(K_blk,1)
    for i in 1:nnodel 
        # |-> ugly as fuck, but faster and more memory friendly than semivectorised original version
        for j in i:nnodel            
            # x-velocity equation (1st, 3rd, 5th,... rows of stiffness matrices)            
            @inbounds @simd for k in 1:nn
                # x-velocity (1st, 3th, 5th,... columns)
                K_blk[k,indx] += ω[k] *
                    (dNdx[k,i]*(η11[k]*dNdx[k,j] + C*η13[k]*dNdy[k,j]) +
                     dNdy[k,i]*(η13[k]*dNdx[k,j] + C*η33[k]*dNdy[k,j]))
            end            
            indx += 1
            @inbounds @simd for k in 1:nn
                # y-velocity equation (2nd, 4th, 6th,... rows of stiffness matrices)
                K_blk[k,indx] += ω[k] *
                    (dNdx[k,i]*(η12[k]*dNdy[k,j] + C*η13[k]*dNdx[k,j]) +
                     dNdy[k,i]*(η23[k]*dNdy[k,j] + C*η33[k]*dNdx[k,j])) 
            end
            indx += 1
        end

        for j in i:nnodel
            if j > i                
                # x-velocity equation (3rd, 5th, 7th... rows of stiffness matrices)            
                @inbounds @simd for k in 1:nn
                    # x-velocity (1st, 3th, 5th,... columns)
                    K_blk[k,indx] += ω[k] * 
                        (dNdx[k,i]*(η13[k]*dNdx[k,j] + C*η33[k]*dNdy[k,j]) +
                         dNdy[k,i]*(η12[k]*dNdx[k,j] + C*η23[k]*dNdy[k,j]))                                
                end
                indx   += 1
            end
            @inbounds @simd for k in 1:nn
                # y-velocity equation (2nd, 4th, 6th,... rows of stiffness matrices)
                K_blk[k,indx] += ω[k] *
                    (dNdx[k,i]*(η23[k]*dNdy[k,j] + C*η33[k]*dNdx[k,j]) +
                     dNdy[k,i]*(η22[k]*dNdy[k,j] + C*η23[k]*dNdx[k,j]))                                 
            end
            indx += 1
        end
        
    end

end

# @inline function _fill_Kblk!(K_blk, 𝓒, Visc_blk, ::Val{Anisotropic}, ω, dNdx, dNdy, nnodel,els,ip) 

#     # Unpack viscous tensor
#     η11 = view(𝓒.η11,els,ip).*Visc_blk
#     η22 = view(𝓒.η33,els,ip).*Visc_blk
#     η33 = view(𝓒.η55,els,ip).*Visc_blk
#     η12 = view(𝓒.η13,els,ip).*Visc_blk
#     η13 = view(𝓒.η15,els,ip).*Visc_blk.*1
#     η23 = view(𝓒.η35,els,ip).*Visc_blk.*1

#     C1 =  2/3; # Used instead of 'D' matrix in standard assembly
#     C2 = -1/3; # Used instead of 'D' matrix in standard assembly
#     C3 =  0.5; # Used instead of 'D' matrix in standard assembly
#     indx = 1
#     nn = size(K_blk,1)
#     for i in 1:nnodel 
#         # |-> ugly as fuck, but faster and more memory friendly than semivectorised original version
#         for j in i:nnodel            
#             # x-velocity equation (1st, 3rd, 5th,... rows of stiffness matrices)            
#             @inbounds @simd for k in 1:nn
#                 # x-velocity (1st, 3th, 5th,... columns)
#                 K_blk[k,indx] += ω[k] *
#                     dNdx[k,i]*( η11[k]*C1*dNdx[k,j] + η12[k]*C2*dNdx[k,j] + η13[k]*(C3*dNdx[k,j])) +
#                     dNdy[k,i]*( η13[k]*C1*dNdx[k,j] + η23[k]*C2*dNdx[k,j] + η33[k]*(C3*dNdx[k,j])) 
#             end            
#             indx   += 1
#             @inbounds @simd for k in 1:nn
#                 # y-velocity equation (2nd, 4th, 6th,... rows of stiffness matrices)
#                 K_blk[k,indx] += ω[k] *

#                     (dNdx[k,i]*(2*η12[k]*dNdy[k,j] + η13[k]*dNdx[k,j])/2 +
#                      dNdy[k,i]*(2*η23[k]*dNdy[k,j] + η33[k]*dNdx[k,j])/2) 


#                     dNdx[k,i]*( η11[k]*C2*dNdy[k,j] + η12[k]*C1*dNdy[k,j] + η13[k]*(C3*dNdy[k,j])) +
#                     dNdy[k,i]*( η13[k]*C2*dNdy[k,j] + η23[k]*C1*dNdy[k,j] + η33[k]*(C3*dNdy[k,j]))
#             end
#             indx   += 1
#         end

#         for j in i:nnodel
#             if j > i                
#                 # x-velocity equation (3rd, 5th, 7th... rows of stiffness matrices)            
#                 @inbounds @simd for k in 1:nn
#                     # x-velocity (1st, 3th, 5th,... columns)
#                     K_blk[k,indx] += ω[k] *
#                         dNdx[k,i]*( η13[k]*C1*dNdx[k,j] + η23[k]*C2*dNdx[k,j] + η33[k]*(C3*dNdx[k,j])) +
#                         dNdy[k,i]*( η12[k]*C1*dNdx[k,j] + η22[k]*C2*dNdx[k,j] + η23[k]*(C3*dNdx[k,j]))
#                 end
#                 indx   += 1
#             end
#             @inbounds @simd for k in 1:nn
#                 # y-velocity equation (2nd, 4th, 6th,... rows of stiffness matrices)
#                 K_blk[k,indx] += ω[k] *
#                     dNdx[k,i]*( η13[k]*C2*dNdy[k,j] + η23[k]*C1*dNdy[k,j] + η33[k]*(C3*dNdy[k,j])) +
#                     dNdy[k,i]*( η12[k]*C2*dNdy[k,j] + η22[k]*C1*dNdy[k,j] + η23[k]*(C3*dNdy[k,j]))
#             end
#             indx += 1
#         end
        
#     end

# end

@inline function _fill_Gblk!(G_blk,NP_blk, ω,dNdx,dNdy, nPdofel,nUdofel,nnodel)
    N = ones(1,nnodel)
    @inbounds  for i in 1:nPdofel        
        tmp1 =  ω .* view(NP_blk,:,i)
        # tmp2 = repeat(tmp1,1,nnodel)
        tmp2 = tmp1.*N
        ii = (i-1)*nUdofel .+ (1:2:nUdofel)
        @views G_blk[:,ii] .+= tmp2 .* dNdx
        ii = ii .+ 1
        @views G_blk[:,ii] .+= tmp2 .* dNdy
    end

end

@inline function _fill_Mblk!(M_blk,Visc_blk,NP_blk, ω, nPdofel)
    # ASSEMBLY OF (1/VISCOSITY)-SCALED MASS MATRICES FOR ALL ELEMENTS IN BLOCK
    weight3 = ω./Visc_blk;
    # "weight" divided by viscosity at integration point in all elements
    indx = 1;
    for i in 1:nPdofel
        @inbounds @simd for j in i:nPdofel
            M_blk[:,indx] .+= weight3.*view(NP_blk,:,i).*view(NP_blk,:,j)
            indx += 1
        end
    end
end

@inline function _fill_Fbblk!(Fb_blk,Fg_blk,NU,
    ω, sin_ip,cos_ip,nUdofel)
    # ASSEMBLY OF BUOYANCY FORCE VECTORS FOR ALL ELEMENTS IN BLOCK
    cte = ω .* Fg_blk * NU
    @views begin
        Fb_blk[:,1:2:nUdofel] .+= sin_ip .* cte
        Fb_blk[:,2:2:nUdofel] .+= cos_ip .* cte
    end
end

@inline function _element_viscosity(η,EL2NOD,PhaseID,els,Nip)
    idx = view(EL2NOD,1:3,els)
    Visc_blk = vec(view(η.val,idx)'*Nip')

    # Cap max and min viscosities
    # Visc_blk = max.(Visc_blk,1e18)
    # Visc_blk = min.(Visc_blk,1e24)

    # Make sure a column-vector is returned
    return  Visc_blk
end

_element_viscosity(η::Isoviscous{T},EL2NOD,PhaseID,els,Nip) where{T} = η.val

@inline function _element_density(ρ,EL2NOD,PhaseID,els,Nip)
    idx = view(EL2NOD,1:3,els)
    Dens_blk = vec(view(ρ,idx)'*Nip')

    return  Dens_blk
end
