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
    EL2DOF[1:ndim:nUdofel,:] .= @. ndim*(EL2NOD[1:nnodel,:]-1) + 1
    EL2DOF[2:ndim:nUdofel,:] .= @. ndim*(EL2NOD[1:nnodel,:]-1) + 2
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
    F0xx_blk,F0zz_blk,F0xz_blk,F0zx_blk = 
        similar(τxx_blk),similar(τxx_blk), similar(τxx_blk),similar(τxx_blk)
    # -- velocity partial derivatives
    ∂Ux∂x,∂Uz∂z,∂Ux∂z,∂Uz∂x =
        similar(τxx_blk),similar(τxx_blk), similar(τxx_blk),similar(τxx_blk)
    # -- Temperature gradient
    ∂T∂x, ∂T∂z = similar(τxx_blk), similar(τxx_blk)

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
            F0xx_blk = deepcopy(Fxx_blk)
            F0xz_blk = deepcopy(Fxz_blk)
            F0zx_blk = deepcopy(Fzx_blk)
            F0zz_blk = deepcopy(Fzz_blk)

            # #=======================================================================
            # PROPERTIES OF ELEMENTS AT ip-TH EVALUATION POINT
            # =======================================================================#
            η_blk = _element_viscosity(η,EL2NOD,PhaseID,il:iu,N_ip)

            #===========================================================================================================
            # CALCULATE 2nd JACOBIAN (FROM CARTESIAN TO POLAR COORDINATES --> curved edges), ITS DETERMINANT AND INVERSE
            # ===========================================================================================================
            # NOTE: For triangular elements with curved edges the Jacobian needs to be computed at each integration
            # point (inside the integration loop). =#
            th_ip = gemmt(VCOORD_th', N_ip')
            r_ip = gemmt(VCOORD_r', N_ip') # VCOORD_r' * N_ip'

            @inbounds  _derivative_weights!(dNds[ip],ω,dNdx,dNdy,w_ip[ip],th_ip,r_ip,sin_ip,cos_ip,
                R_21,R_31,Th_21,Th_31, detJ_PL, invJx_double, invJz_double)
                
            _velocityderivatives!(∂Ux∂x, ∂Uz∂z, ∂Ux∂z, ∂Uz∂x,
                                  εxx_blk, εzz_blk, εxz_blk,
                                  B, dNdx, dNdy,U_blk) 

            _stress!(τxx_blk, τzz_blk, τxz_blk,
                     εxx_blk, εzz_blk, εxz_blk,
                     η_blk, 𝓒, Val(η),
                     il:iu,ip)
            
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

@inline function _Fij_Rk4!( Fxx_blk, Fzz_blk, Fxz_blk, Fzx_blk,
    ∂Ux∂x, ∂Uz∂z, ∂Ux∂z, ∂Uz∂x,
    Δt)

    one_sixth = 1/6
    @turbo for i in axes(Fxx_blk,1)
       # cache them out
        ∂Ux∂xᵢ, ∂Ux∂zᵢ, ∂Uz∂xᵢ, ∂Uz∂zᵢ = ∂Ux∂x[i], ∂Ux∂z[i], ∂Uz∂x[i], ∂Uz∂z[i]
        Fxx, Fxz, Fzx, Fzz = Fxx_blk[i], Fxz_blk[i], Fzx_blk[i], Fzz_blk[i]
        # 1st step
        k1xx = Δt*(∂Ux∂xᵢ*Fxx + ∂Ux∂zᵢ*Fzx)  
        k1xz = Δt*(∂Ux∂xᵢ*Fxz + ∂Ux∂zᵢ*Fzz)  
        k1zx = Δt*(∂Uz∂xᵢ*Fxx + ∂Uz∂zᵢ*Fzx)  
        k1zz = Δt*(∂Uz∂xᵢ*Fxz + ∂Uz∂zᵢ*Fzz)  
        # 2nd step
        Fxxi = (Fxx + k1xx*0.5)
        Fxzi = (Fxz + k1xz*0.5)
        Fzxi = (Fzx + k1zx*0.5)
        Fzzi = (Fzz + k1zz*0.5)
        k2xx = Δt*(∂Ux∂xᵢ*Fxxi + ∂Ux∂zᵢ*Fzxi)
        k2xz = Δt*(∂Ux∂xᵢ*Fxzi + ∂Ux∂zᵢ*Fzzi)
        k2zx = Δt*(∂Uz∂xᵢ*Fxxi + ∂Uz∂zᵢ*Fzxi)
        k2zz = Δt*(∂Uz∂xᵢ*Fxzi + ∂Uz∂zᵢ*Fzzi)
        # 3rd step
        Fxxi = Fxx + k2xx*0.5
        Fxzi = Fxz + k2xz*0.5
        Fzxi = Fzx + k2zx*0.5
        Fzzi = Fzz + k2zz*0.5
        k3xx = Δt*(∂Ux∂xᵢ*Fxxi + ∂Ux∂zᵢ*Fzxi)
        k3xz = Δt*(∂Ux∂xᵢ*Fxzi + ∂Ux∂zᵢ*Fzzi)
        k3zx = Δt*(∂Uz∂xᵢ*Fxxi + ∂Uz∂zᵢ*Fzxi)
        k3zz = Δt*(∂Uz∂xᵢ*Fxzi + ∂Uz∂zᵢ*Fzzi)
        # 4th step
        Fxxi = Fxx + k3xx
        Fxzi = Fxz + k3xz
        Fzxi = Fzx + k3zx
        Fzzi = Fzz + k3zz
        k4xx = Δt*(∂Ux∂xᵢ*Fxxi + ∂Ux∂zᵢ*Fzxi)
        k4xz = Δt*(∂Ux∂xᵢ*Fxzi + ∂Ux∂zᵢ*Fzzi)
        k4zx = Δt*(∂Uz∂xᵢ*Fxxi + ∂Uz∂zᵢ*Fzxi)
        k4zz = Δt*(∂Uz∂xᵢ*Fxzi + ∂Uz∂zᵢ*Fzzi)
        # last step
        Fxz_blk[i] += (k1xz + 2*(k2xz + k3xz) + k4xz)*one_sixth
        Fxx_blk[i] += (k1xx + 2*(k2xx + k3xx) + k4xx)*one_sixth
        Fzx_blk[i] += (k1zx + 2*(k2zx + k3zx) + k4zx)*one_sixth
        Fzz_blk[i] += (k1zz + 2*(k2zz + k3zz) + k4zz)*one_sixth
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
    @inbounds @simd for i in els
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
