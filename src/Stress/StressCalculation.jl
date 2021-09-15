struct SymmetricTensor{T} 
    xx::Matrix{T}
    zz::Matrix{T}
    xz::Matrix{T}
end

struct Gradient{T}
    ∂x::Matrix{T}
    ∂z::Matrix{T}
end

function initstress(nel)
    F   = [@SMatrix [1.0 0.0; 0.0 1.0]  for _ in 1:nel, _ in 1:6]
    F0  = deepcopy(F)
    τ   = SymmetricTensor(fill(0.0,nel,6),fill(0.0,nel,6),fill(0.0,nel,6))
    ε   = SymmetricTensor(fill(0.0,nel,6),fill(0.0,nel,6),fill(0.0,nel,6))
    ∇T  = Gradient(fill(0.0,nel,6), fill(0.0,nel,6))
    return F, F0, τ, ε, ∇T
end

function _stress!(
    F,
    U,
    iel,
    DoF_U,
    θ,
    r,
    nip,
    SF_Stokes::ShapeFunctionsStokes,
    Δt
)
    # g, 
    # ρ, 
    # η, 
    # 𝓒,

    N, ∇N, NP, dN3ds, w_ip, N3 = 
        SF_Stokes.N, SF_Stokes.∇N, SF_Stokes.NP, SF_Stokes.dN3ds, SF_Stokes.w_ip, SF_Stokes.N3 
    
    # Polar coordinates of element nodes
    θ_el = @SVector [θ[i, iel] for i in 1:3]
    r_el = @SVector [r[i, iel] for i in 1:3]
    coords = SMatrix{3,2}([θ_el r_el])

    # Jacobian n. 1 (p:=polar, l:=local): reference element --> current element
    J_pl = dN3ds * coords
    detJ_pl = mydet(J_pl)

    # the Jacobian ∂ξ∂θ to transform local (ξ, η) into global (θ,r) derivatives
    #     ∂ξ∂θ = [ R_31    -R_21
    #             -Th_31    Th_21] / detJa_PL
    R_21 = r_el[2] - r_el[1]  # = -detJa_PL*deta_dth
    R_31 = r_el[3] - r_el[1]  # =  detJa_PL*dxi_dth
    Th_31 = θ_el[3] - θ_el[1] # = -detJa_PL*dxi_dr
    Th_21 = θ_el[2] - θ_el[1] # =  detJa_PL*deta_dr

    Udofs = DoF_U[iel]
    U_el = @SVector [
        U[Udofs[1]], U[Udofs[2]], U[Udofs[3]], U[Udofs[4]],  U[Udofs[5]],  U[Udofs[6]],
        U[Udofs[7]], U[Udofs[8]], U[Udofs[9]], U[Udofs[10]], U[Udofs[11]], U[Udofs[12]]
    ]

    # INTEGRATION LOOP
    @inbounds for ip in 1:nip

        # Unpack shape functions 
        # N_ip = N[ip]
        # NP_ip = NP[ip]
        N3_ip = N3[ip]
        ∇N_ip = ∇N[ip]

        # ρ at ith integration point
        # ρ_ip = mydot(ρ_el, N3_ip)
        # η_ip = _element_viscosity(η, gr.e2n, PhaseID, iel, N3_ip)

        # Polar coordinates of the integration points
        θ_ip = mydot(θ_el, N3_ip)
        r_ip = mydot(r_el, N3_ip)
        sin_ip, cos_ip = sincos(θ_ip)
        cos_ip_r_ip = cos_ip / r_ip
        sin_ip_r_ip = sin_ip / r_ip

        # Build inverse of the 2nd Jacobian
        # invJ_double = @SMatrix [(∂ξ∂θ[1,1]*cos_ip_r_ip-∂ξ∂θ[2,1]*sin_ip) (-∂ξ∂θ[1,2]*cos_ip_r_ip+∂ξ∂θ[2,2]*sin_ip); 
        #                        -(∂ξ∂θ[1,1]*sin_ip_r_ip+∂ξ∂θ[2,1]*cos_ip) (∂ξ∂θ[1,2]*sin_ip_r_ip+∂ξ∂θ[2,2]*cos_ip) ]
        invJ_double = @SMatrix [
             R_31*cos_ip_r_ip-Th_31*sin_ip   -R_21*cos_ip_r_ip+Th_21*sin_ip
            -R_31*sin_ip_r_ip-Th_31*cos_ip    R_21*sin_ip_r_ip+Th_21*cos_ip
        ]

        # Partial derivatives
        ∂N∂x = invJ_double * ∇N_ip / detJ_pl

        # Update elemental matrices
        # D = DMatrix(𝓒, iel, ip, Val(η)) 
        # Use this to compute directly strain rate (i.e. B*Uel)
        # B = @SMatrix [
        #     ∂N∂x[1,1]   0           ∂N∂x[1,2]   0           ∂N∂x[1,3]   0           ∂N∂x[1,4]   0           ∂N∂x[1,5]   0           ∂N∂x[1,6]   0
        #     0           ∂N∂x[2,1]   0           ∂N∂x[2,2]   0           ∂N∂x[2,3]   0           ∂N∂x[2,4]   0           ∂N∂x[2,5]   0           ∂N∂x[2,6]
        #     ∂N∂x[2,1]   ∂N∂x[1,1]   ∂N∂x[2,2]   ∂N∂x[1,2]   ∂N∂x[2,3]   ∂N∂x[1,3]   ∂N∂x[2,4]   ∂N∂x[1,4]   ∂N∂x[2,5]   ∂N∂x[1,5]   ∂N∂x[2,6]   ∂N∂x[1,6]
        # ]

        # modified B to calculcate partial derivatives of velocity
        B = @SMatrix [
            ∂N∂x[1,1]   0.0         ∂N∂x[1,2]   0.0         ∂N∂x[1,3]   0.0         ∂N∂x[1,4]   0.0         ∂N∂x[1,5]   0.0         ∂N∂x[1,6]   0.0
            0.0         ∂N∂x[2,1]   0.0         ∂N∂x[2,2]   0.0         ∂N∂x[2,3]   0.0         ∂N∂x[2,4]   0.0         ∂N∂x[2,5]   0.0         ∂N∂x[2,6]
            ∂N∂x[2,1]   0.0         ∂N∂x[2,2]   0.0         ∂N∂x[2,3]   0.0         ∂N∂x[2,4]   0.0         ∂N∂x[2,5]   0.0         ∂N∂x[2,6]   0.0        
            0.0         ∂N∂x[1,1]   0.0         ∂N∂x[1,2]   0.0         ∂N∂x[1,3]   0.0         ∂N∂x[1,4]   0.0         ∂N∂x[1,5]   0.0         ∂N∂x[1,6]
        ]

        ∂U∂x = B*U_el # [∂Ux∂x ∂Uz∂z ∂Ux∂z ∂Uz∂x]
        # transpose of the velocity gradient
        ∇Uᵀ = @SMatrix [
            ∂U∂x[1] ∂U∂x[4]
            ∂U∂x[3] ∂U∂x[2]
        ]

        # F0 = F[iel, ip]
        k1 = Δt * ∇Uᵀ * F[iel, ip]
        
        Fi = k1*0.5 .+ F[iel, ip]

        k2 = Δt * ∇Uᵀ * Fi
        
        Fi = k2*0.5 .+ F[iel, ip]

        k3 = Δt * ∇Uᵀ * Fi
        
        Fi = k3 .+ F[iel, ip]
        k4 = Δt * ∇Uᵀ * Fi

        F[iel, ip] = F[iel, ip] + (k1 + 2*(k2 + k3) +k4)/6
        
    end

end

s_F_Rk4(∇Uᵀ, F) = ∇Uᵀ*F

function stress!(F, U, nel, DoF_U, coordinates, nip, SF_Stokes, Δt)
    @batch per=core for iel in 1:nel
        # @code_warntype
        _stress!(F, U, iel, DoF_U, coordinates.θ, coordinates.r, nip, SF_Stokes, Δt)
    end
end

function stress(U, T, F, 𝓒, τ, ε, EL2NOD, theta, r, η, PhaseID, Δt)
    Uscaled = U

    # ============================================ MODEL AND BLOCKING PARAMETERS
    ndim = 2
    # nvert = 3
    nip = 6
    nnodel = size(EL2NOD,1)
    nnodel = 6
    nel = size(EL2NOD,2)
    nUdofel = ndim*nnodel
    EL2DOF = Array{Int32}(undef,nUdofel,nel)
    @views EL2DOF[1:ndim:nUdofel,:] .= @. ndim*(EL2NOD[1:nnodel,:]-1) + 1
    @views EL2DOF[2:ndim:nUdofel,:] .= @. ndim*(EL2NOD[1:nnodel,:]-1) + 2
    nelblk = min(nel, 1200)
    nblk = ceil(Int,nel/nelblk)
    il = one(nelblk); iu = nelblk
    # =========== PREPARE INTEGRATION POINTS & DERIVATIVES wrt LOCAL COORDINATES   
    ni, nn, nn3 = Val(nip),Val(nnodel), Val(3)
    _,dNds,_ ,w_ip = _get_SF(ni,nn)
    N3, _,dN3ds,_ = _get_SF(ni,nn3)
    # ============================================================= ALLOCATIONS
    detJ_PL = Vector{Float64}(undef,nelblk)
    R_31 = similar(detJ_PL) # =  detJa_PL*dxi_dth
    R_21 = similar(detJ_PL) # = -detJa_PL*deta_dth
    Th_31 = similar(detJ_PL) # = -detJa_PL*dxi_dr
    Th_21 = similar(detJ_PL) # =  detJa_PL*deta_dr
    dNdx = Matrix{Float64}(undef,nelblk,size(dNds[1],2))
    dNdy = similar(dNdx)
    ω = Vector{Float64}(undef,Int(nelblk))
    invJx_double = Matrix{Float64}(undef,Int(nelblk),ndim) # storage for x-components of Jacobi matrix
    invJz_double = similar(invJx_double)  # storage for z-components of Jacobi matrix  
    # ==================================== STORAGE FOR DATA OF ELEMENTS IN BLOCK
    B = fill(0.0,nelblk,nUdofel)
    # -- stress tensor
    τxx_blk = Vector{Float64}(undef,nelblk)
    τzz_blk,τxz_blk = similar(τxx_blk), similar(τxx_blk)
    # -- strain tensor
    εxx_blk,εzz_blk,εxz_blk =
        similar(τxx_blk),similar(τxx_blk), similar(τxx_blk)
    # -- deformation gradient 
    Fxx_blk,Fzz_blk,Fxz_blk,Fzx_blk = 
        similar(τxx_blk),similar(τxx_blk), similar(τxx_blk),similar(τxx_blk)
    # -- deformation gradient 
    # F0xx_blk,F0zz_blk,F0xz_blk,F0zx_blk = 
    #     similar(τxx_blk),similar(τxx_blk), similar(τxx_blk),similar(τxx_blk)
    # -- velocity partial derivatives
    ∂Ux∂x,∂Uz∂z,∂Ux∂z,∂Uz∂x =
        similar(τxx_blk),similar(τxx_blk), similar(τxx_blk),similar(τxx_blk)
    # -- Temperature gradient
    # ∂T∂x, ∂T∂z = similar(τxx_blk), similar(τxx_blk)

    #=========================================================================
    BLOCK LOOP - MATRIX COMPUTATION
    =========================================================================#
    for ib in 1:nblk
        #===========================================================================
        CALCULATE JACOBIAN, ITS DETERMINANT AND INVERSE
        ===========================================================================#
        """
        NOTE: For triangular elements with non-curved edges the Jacobian is
              the same for all integration points (i.e. calculated once
              before the integration loop). Further, linear 3-node shape are
              sufficient to calculate the Jacobian.
        """
        VCOORD_th = view(theta,:,il:iu)
        VCOORD_r = view(r,:,il:iu)

        J_th = gemm(VCOORD_th', dN3ds')
        J_r = gemm(VCOORD_r', dN3ds')

        _fill_R_J!(J_th,J_r,VCOORD_th,VCOORD_r,
            R_31,R_21,Th_31,Th_21,detJ_PL)

        idx = view(EL2DOF,:,il:iu)
        U_blk = transpose(view(Uscaled, idx))

        sin_ip = similar(detJ_PL)
        cos_ip = similar(detJ_PL)

        for ip=1:nip
            
            @inbounds N_ip = N3[ip]
            _getblock!(Fxx_blk,F,il:iu,ip,1,1)
            _getblock!(Fxz_blk,F,il:iu,ip,1,2)
            _getblock!(Fzx_blk,F,il:iu,ip,2,1)
            _getblock!(Fzz_blk,F,il:iu,ip,2,2)

            # #=======================================================================
            # PROPERTIES OF ELEMENTS AT ip-TH EVALUATION POINT
            # =======================================================================#
            # η_blk = _element_viscosity(η,EL2NOD,PhaseID,il:iu,N_ip)

            #============================================================================================================
            # CALCULATE 2nd JACOBIAN (FROM CARTESIAN TO POLAR COORDINATES --> curved edges), ITS DETERMINANT AND INVERSE
            # ===========================================================================================================
            # NOTE: For triangular elements with curved edges the Jacobian needs to be computed at each integration
            # point (inside the integration loop). =#
            th_ip = gemmt(VCOORD_th', N_ip')
            r_ip = gemmt(VCOORD_r', N_ip') # VCOORD_r' * N_ip'

            @inbounds _derivative_weights!(dNds[ip],ω,dNdx,dNdy,w_ip[ip],th_ip,r_ip,sin_ip,cos_ip,
                R_21,R_31,Th_21,Th_31, detJ_PL, invJx_double, invJz_double)
                
            _velocityderivatives!(∂Ux∂x, ∂Uz∂z, ∂Ux∂z, ∂Uz∂x,
                                  εxx_blk, εzz_blk, εxz_blk,
                                  B, dNdx, dNdy,U_blk) 

            # _stress!(τxx_blk, τzz_blk, τxz_blk,
            #          εxx_blk, εzz_blk, εxz_blk,
            #          η_blk, 𝓒, Val(η),
            #          il:iu,ip)
            
            _Fij_Rk4!(Fxx_blk, Fzz_blk, Fxz_blk, Fzx_blk,
                      ∂Ux∂x, ∂Uz∂z, ∂Ux∂z, ∂Uz∂x,
                      Δt)

            _fillstress!(F, τ, ε,
                         Fxx_blk, Fzz_blk, Fxz_blk, Fzx_blk,
                         τxx_blk, τzz_blk, τxz_blk,
                         εxx_blk, εzz_blk, εxz_blk,
                         il:iu,ip)

            # _∇T!(∂T∂x, ∂T∂z, dNdx, dNdy, T_blk)   

            # _fill∇T!(∇T, ∂T∂x, ∂T∂z, il:iu, ip)

        end # end integration point loop
               
        # -------------------------------- READJUST START, END AND SIZE OF BLOCK
        il += nelblk
        if ib == nblk-1
            # ------------ Account for different number of elements in last block
            nelblk = nel - il + 1  # new block size
            # --------------------------------- Reallocate at blocks at the edge 
            detJ_PL = Vector{Float64}(undef,nelblk)
            R_31 = similar(detJ_PL) 
            R_21 = similar(detJ_PL) 
            Th_31 = similar(detJ_PL) 
            Th_21 = similar(detJ_PL) 
            dNdx = Matrix{Float64}(undef,nelblk,size(dNds[1],2))
            dNdy = similar(dNdx)
            ω = Vector{Float64}(undef,Int(nelblk))
            invJx_double = Matrix{Float64}(undef,Int(nelblk),ndim) 
            invJz_double = similar(invJx_double) 
            B = fill(0.0,nelblk,nUdofel)
                # -- stress tensor
            τxx_blk = Vector{Float64}(undef,nelblk)
            τzz_blk,τxz_blk = similar(τxx_blk), similar(τxx_blk)
                # -- strain tensor
            εxx_blk,εzz_blk,εxz_blk =
               similar(τxx_blk),similar(τxx_blk), similar(τxx_blk)        
                # -- deformation gradient 
            Fxx_blk,Fzz_blk,Fxz_blk,Fzx_blk = 
               similar(τxx_blk),similar(τxx_blk), similar(τxx_blk),similar(τxx_blk)
                # -- velocity partial derivatives
            ∂Ux∂x,∂Uz∂z,∂Ux∂z,∂Uz∂x =
               similar(τxx_blk),similar(τxx_blk), similar(τxx_blk),similar(τxx_blk)
        end
        iu += nelblk

    end # end block loop
    
    #===========================================================================
    Tensor second invariants
    ===========================================================================#
    τII = secondinvariant(τ)
    εII = secondinvariant(ε)    

    return F, τ, ε, τII, εII
    
end # END OF ASSEMBLY FUNCTION

function updateF!(F,Fxx,Fzz,Fxz,Fzx)
    Threads.@threads for j in axes(Fxx,2)    
        @inbounds for i in axes(Fxx,1)
            F[i,j] = @SMatrix [Fxx[i,j] Fxz[i,j]; Fzx[i,j] Fzz[i,j]]
        end
    end
end

@inline function _velocityderivatives!( ∂Ux∂x, ∂Uz∂z, ∂Ux∂z, ∂Uz∂x,
                                εxx_blk, εzz_blk, εxz_blk,
                                B, dNdx, dNdy, U_blk)        

    Ux = @views U_blk[:,1:2:end-1]
    Uz = @views U_blk[:,2:2:end]
    # -- Strain rate tensor
    rowdot!(εxx_blk, dNdx, Ux)
    rowdot!(εzz_blk, dNdy, Uz)

    c = Int32(0)
    for j in 1:2:size(B,2)-1
        c += one(c)
        for i in axes(dNdy,1)
            @inbounds B[i,j] = dNdy[i,c]
            @inbounds B[i,j+1] = dNdx[i,c]
        end        
    end

    rowdot!(εxz_blk,B,U_blk) 
    εxz_blk .*= 0.5

    # -- Partial derivatives
    ∂Ux∂x .= εxx_blk
    ∂Uz∂z .= εzz_blk
    rowdot!(∂Ux∂z,dNdy, Ux)
    rowdot!(∂Uz∂x,dNdx, Uz)

end


@inline function _∇T!(∂T∂x, ∂T∂z, dNdx, dNdy, T_blk)        
    # -- Temperature gradient
    rowdot!(∂T∂x, dNdx, T_blk)
    rowdot!(∂T∂z, dNdy, T_blk)
end

@inline function _fill∇T!(∇T, ∂T∂x, ∂T∂z, els, ip)
    @inbounds for (c,i) in enumerate(els)
        ∇T.∂x[i,ip] = ∂T∂x[c]
        ∇T.∂z[i,ip] = ∂T∂z[c]
    end
end

@inline function _stress!(  τxx_blk, τzz_blk, τxz_blk,
                            εxx_blk, εzz_blk, εxz_blk,
                            η_blk, 𝓒, ::Val{Isotropic},
                            els,ip)
    # C1, C2 = 4/3, -2/3
    C1, C2 = 2/3, -1/3
    @inbounds @simd for i in axes(τxx_blk,1)
        τxx_blk[i] =  C1 * η_blk[i] * εxx_blk[i] + C2 * η_blk[i] * εzz_blk[i] 
        τzz_blk[i] =  C2 * η_blk[i] * εxx_blk[i] + C1 * η_blk[i] * εzz_blk[i] 
        τxz_blk[i] =   2 * η_blk[i] * εxz_blk[i]
    end

end

@inline function _stress!(  τxx_blk, τzz_blk, τxz_blk,
                            εxx_blk, εzz_blk, εxz_blk,
                            η_blk, 𝓒, ::Val{Anisotropic},
                            els,ip)

     # Unpack viscous tensor
     η11 = view(𝓒.η11,els,ip).*η_blk
     η22 = view(𝓒.η33,els,ip).*η_blk
     η33 = view(𝓒.η55,els,ip).*η_blk
     η12 = view(𝓒.η13,els,ip).*η_blk
    #  η13 = 0*view(𝓒.η15,els,ip).*η_blk
    #  η23 = 0*view(𝓒.η35,els,ip).*η_blk

    @turbo for i in axes(τxx_blk,1)
        τxx_blk[i] = η11[i] * εxx_blk[i] + η12[i] * εzz_blk[i] 
        τzz_blk[i] = η12[i] * εxx_blk[i] + η22[i] * εzz_blk[i] 
        τxz_blk[i] = η33[i] * εxz_blk[i]
    end

end

@inline function _Fij!( Fxx_blk, Fzz_blk, Fxz_blk, Fzx_blk,
                        ∂Ux∂x, ∂Uz∂z, ∂Ux∂z, ∂Uz∂x,
                        Δt)
    
    @turbo for i in eachindex(Fxx_blk)
        Fxx_blk[i] += Δt * (∂Ux∂x[i]*Fxx_blk[i] + ∂Ux∂z[i] * Fzx_blk[i])
        Fxz_blk[i] += Δt * (∂Ux∂x[i]*Fxz_blk[i] + ∂Ux∂z[i] * Fzz_blk[i])
        Fzx_blk[i] += Δt * (∂Uz∂x[i]*Fxx_blk[i] + ∂Uz∂z[i] * Fzx_blk[i])
        Fzz_blk[i] += Δt * (∂Uz∂x[i]*Fxz_blk[i] + ∂Uz∂z[i] * Fzz_blk[i])
    end

end

@inline function _Fij_Rk4(Δt, ∂Ux∂xᵢ, ∂Ux∂zᵢ, ∂Uz∂xᵢ, ∂Uz∂zᵢ, Fxx, Fxz, Fzx, Fzz)
    kxx = @muladd Δt*(∂Ux∂xᵢ*Fxx + ∂Ux∂zᵢ*Fzx)
    kxz = @muladd Δt*(∂Ux∂xᵢ*Fxz + ∂Ux∂zᵢ*Fzz)
    kzx = @muladd Δt*(∂Uz∂xᵢ*Fxx + ∂Uz∂zᵢ*Fzx)
    kzz = @muladd Δt*(∂Uz∂xᵢ*Fxz + ∂Uz∂zᵢ*Fzz)
    return kxx, kxz, kzx, kzz
end

@inline function _Fij_Rk4!( 
    Fxx_blk, Fzz_blk, Fxz_blk, Fzx_blk,
    ∂Ux∂x, ∂Uz∂z, ∂Ux∂z, ∂Uz∂x,
    Δt
)
    
    one_sixth = 1/6
    @turbo for i in axes(Fxx_blk,1)
       # cache them out
        ∂Ux∂xᵢ, ∂Ux∂zᵢ, ∂Uz∂xᵢ, ∂Uz∂zᵢ = ∂Ux∂x[i], ∂Ux∂z[i], ∂Uz∂x[i], ∂Uz∂z[i]
        Fxx, Fxz, Fzx, Fzz = Fxx_blk[i], Fxz_blk[i], Fzx_blk[i], Fzz_blk[i]
        # 1st step
        k1xx, k1xz, k1zx, k1zz = _Fij_Rk4(Δt, ∂Ux∂xᵢ, ∂Ux∂zᵢ, ∂Uz∂xᵢ, ∂Uz∂zᵢ, Fxx, Fxz, Fzx, Fzz)
        # 2nd step
        Fxxi = fma(k1xx, 0.5, Fxx)
        Fxzi = fma(k1xz, 0.5, Fxz)
        Fzxi = fma(k1zx, 0.5, Fzx)
        Fzzi = fma(k1zz, 0.5, Fzz)
        k2xx, k2xz, k2zx, k2zz = _Fij_Rk4(Δt, ∂Ux∂xᵢ, ∂Ux∂zᵢ, ∂Uz∂xᵢ, ∂Uz∂zᵢ, Fxxi, Fxzi, Fzxi, Fzzi)
        # 3rd step
        Fxxi = fma(k2xx, 0.5, Fxx)
        Fxzi = fma(k2xz, 0.5, Fxz)
        Fzxi = fma(k2zx, 0.5, Fzx)
        Fzzi = fma(k2zz, 0.5, Fzz)
        k3xx, k3xz, k3zx, k3zz = _Fij_Rk4(Δt, ∂Ux∂xᵢ, ∂Ux∂zᵢ, ∂Uz∂xᵢ, ∂Uz∂zᵢ, Fxxi, Fxzi, Fzxi, Fzzi)
        # 4th step
        Fxxi = Fxx + k3xx
        Fxzi = Fxz + k3xz
        Fzxi = Fzx + k3zx
        Fzzi = Fzz + k3zz
        k4xx, k4xz, k4zx, k4zz = _Fij_Rk4(Δt, ∂Ux∂xᵢ, ∂Ux∂zᵢ, ∂Uz∂xᵢ, ∂Uz∂zᵢ, Fxxi, Fxzi, Fzxi, Fzzi)
        # last step
        Fxz_blk[i] += @muladd (k1xz + 2*(k2xz + k3xz) + k4xz)*one_sixth
        Fxx_blk[i] += @muladd (k1xx + 2*(k2xx + k3xx) + k4xx)*one_sixth
        Fzx_blk[i] += @muladd (k1zx + 2*(k2zx + k3zx) + k4zx)*one_sixth
        Fzz_blk[i] += @muladd (k1zz + 2*(k2zz + k3zz) + k4zz)*one_sixth
    end

end

secondinvariant(xx,zz,xz) = sqrt( 0.5*(xx^2 + zz^2) + xz^2 ) 

@inline function secondinvariant(A::SymmetricTensor)
    m, n = size(A.xx)
    II  = Matrix{Float64}(undef,m,n)
    xx, zz, xz = A.xx, A.zz, A.xz
    @turbo for i in eachindex(A.xx)
        II[i] = √( 0.5*(xx[i]^2 + zz[i]^2) + xz[i]^2) 
    end
    return II
end

@inline function _fillstress!(F, τ, ε,
                              Fxx_blk, Fzz_blk, Fxz_blk, Fzx_blk,
                              τxx_blk, τzz_blk, τxz_blk,
                              εxx_blk, εzz_blk, εxz_blk,
                              els,ip )
    c = Int32(0)
    @inbounds for i in els
        c += one(c)
        # -- Deformation gradient Fij
        F[i,ip] = @SMatrix [Fxx_blk[c] Fxz_blk[c]; Fzx_blk[c] Fzz_blk[c]]
        # -- Deviatoric strain tensor εij
        ε.xx[i,ip] = εxx_blk[c]
        ε.zz[i,ip] = εzz_blk[c]
        ε.xz[i,ip] = εxz_blk[c]
        # -- Deviatoric stress tensor τij
        τ.xx[i,ip] = τxx_blk[c]
        τ.zz[i,ip] = τzz_blk[c]
        τ.xz[i,ip] = τxz_blk[c]
    end

end

function shearheating(τ::SymmetricTensor{T}, ε::SymmetricTensor{T}) where {T}
    shear_heating = fill(0.0, size(τ.xx))
    # trace = @. (ε.xx + ε.zz)/3
    @avx for i in CartesianIndices(shear_heating) # faster than threading for current matrices size
        shear_heating[i] = τ.xx[i]*(ε.xx[i]- (ε.xx[i] + ε.zz[i])/3) +
                           τ.zz[i]*(ε.zz[i]- (ε.xx[i] + ε.zz[i])/3) +
                           2τ.xz[i]*ε.xz[i]
    end
    shear_heating
end

function rowdot!(C,A,B) 
    @inbounds for i in 1:size(A,1) 
        C[i] = mydot(view(A,i,:),view(B,i,:))
    end
end

function _getblock!(Ablk,A,els,ip,ii,jj)
    c = Int32(0)
    @inbounds for row in els
        c += one(c)
        Ablk[c] = A[row,ip][ii,jj]
    end
end
