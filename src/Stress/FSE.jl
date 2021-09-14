struct FiniteStrainEllipsoid{T}
    x1::T
    x2::T
    y1::T
    y2::T
    a1::T
    a2::T
end

rebuild_FSE(vx1, vx2, vy1, vy2, a1, a2) = [FiniteStrainEllipsoid(vx1[i], vx2[i], vy1[i], vy2[i], a1[i], a2[i]) for i in CartesianIndices(a1)]

function isotropic_lithosphere!(F,idx)
    Is = @SMatrix [1.0 0.0; 0.0 1.0]
    Threads.@threads for i in idx
        @inbounds F[i] = Is
    end
end

function getFSE(F, FSE)
    @batch for iel in eachindex(F)
        local_FSE!(FSE, F, iel)
    end
    FSE
end 

function local_FSE!(FSE, F, iel)
    # # Compute FSE
    # eigval, evect = eigen(F[iel] * F[iel]')
    # # Fill FSE
    # @inbounds FSE[iel] = FiniteStrainEllipsoid(
    #     evect[1,2]::Float64, # vx1
    #     evect[1,1]::Float64, # vx2
    #     evect[2,2]::Float64, # vy1
    #     evect[2,1]::Float64, # vy2
    #     √(abs(eigval[2]))::Float64, # a1 
    #     √(abs(eigval[1]))::Float64, # a2
    # )

    # Compute FSE
    eigval, evect = eigen(F[iel] * F[iel]')
    # Fill FSE
    imax = eigval[2] > eigval[1] ? 2 : 1
    imin = imax == 2 ? 1 : 2
    @inbounds FSE[iel] = FiniteStrainEllipsoid(
        evect[1,imax]::Float64, # vx1
        evect[1,imin]::Float64, # vx2
        evect[2,imax]::Float64, # vy1
        evect[2,imin]::Float64, # vy2
        √(abs(eigval[imax]))::Float64, # a1 
        √(abs(eigval[imin]))::Float64, # a2
    )
end

function FSEips(U, particle_fields, particle_weights, particle_info, gr, coordinates, Δt)
    
    EL2NOD, θ, r = gr.e2n, coordinates.θ, coordinates.r
    Uscaled = U.*1

    # ============================================ MODEL AND BLOCKING PARAMETERS
    ndim = 2
    nip = 6
    nnodel = size(EL2NOD,1)
    nnodel = 3
    nel = size(EL2NOD,2)
    nUdofel = ndim*nnodel
    EL2DOF = Array{Int32}(undef,nUdofel,nel)
    EL2DOF[1:ndim:nUdofel,:] .= @. ndim*(EL2NOD[1:nnodel,:]-1) + 1
    EL2DOF[2:ndim:nUdofel,:] .= @. ndim*(EL2NOD[1:nnodel,:]-1) + 2
    
    # =========== PREPARE INTEGRATION POINTS & DERIVATIVES wrt LOCAL COORDINATES   
    ni, _, nn3 = Val(nip),Val(nnodel), Val(3)
    _,_,dN3ds,_ = _get_SF(ni,nn3)
    
    npart = length(particle_weights)
    
    #=========================================================================
    BLOCK LOOP - MATRIX COMPUTATION
    =========================================================================#
    @inbounds for ipart in 1:npart
        #===========================================================================
        CALCULATE JACOBIAN, ITS DETERMINANT AND INVERSE
        ===========================================================================#
        #=
        NOTE: For triangular elements with non-curved edges the Jacobian is
              the same for all integration points (i.e. calculated once
              before the integration loop). Further, linear 3-node shape are
              sufficient to calculate the Jacobian.
        =#

        # Parent element
        iel = particle_info[ipart].t_parent

        θ_el = @SVector [θ[i, iel] for i in 1:3]
        r_el = @SVector [r[i, iel] for i in 1:3]
        coords = SMatrix{3,2}([θ_el r_el])
    
        # Jacobian n. 1 (p:=polar, l:=local): reference element --> current element
        J_pl = dN3ds * coords
        detJ_pl = det(J_pl) # fast |-> unrolled
        
        # the Jacobian ∂ξ∂θ to transform local (ξ, η) into global (θ,r) derivatives
        #     ∂ξ∂θ = [ R_31    -R_21
        #             -Th_31    Th_21] / detJa_PL
        ∂ξ∂θ = @SMatrix [(r_el[3]-r_el[1])/detJ_pl -(r_el[2]-r_el[1])/detJ_pl
                        -(θ_el[3]-θ_el[1])/detJ_pl  (θ_el[2]-θ_el[1])/detJ_pl]
    
        idx = view(EL2DOF, :, iel)
        U_blk = view(Uscaled, idx)'

        U1 = @SVector [U_blk[1],  U_blk[2]] # (Ux, Uz) node 1
        U2 = @SVector [U_blk[3],  U_blk[4]] # (Ux, Uz) node 2
        U3 = @SVector [U_blk[5],  U_blk[6]] # (Ux, Uz) node 3
        Ux = @SVector [U_blk[1],  U_blk[3], U_blk[5]] # x-vel
        Uz = @SVector [U_blk[2],  U_blk[4], U_blk[6]] # z-vel

        F = @SMatrix [particle_fields.Fxx[ipart]    particle_fields.Fxz[ipart] 
                      particle_fields.Fzx[ipart]    particle_fields.Fzz[ipart]]

        # compute shape functions 
        λ = local_coordinates(coords,  particle_info[ipart].CPolar)

        N3, ∇N = sf_N_tri3(λ[1], λ[2]), sf_dN_tri3(λ[1], λ[2])

        # Polar coordinates of the integration points
        θ_ip = mydot(θ_el, N3)
        r_ip = mydot(r_el, N3)
        cos_ip = cos(θ_ip)
        sin_ip = sin(θ_ip)
        
        # Build inverse of the 2nd Jacobian
        transformation = @SMatrix [cos_ip/r_ip sin_ip; 
                                  -sin_ip/r_ip cos_ip]
        invJ_double = transformation*∂ξ∂θ
        
        # Partial derivatives
        ∂N∂x = invJ_double*∇N
        # B Matrix
        B1 = @SMatrix [∂N∂x[1,1]  0.0
                       0.0        ∂N∂x[2,1]
                       ∂N∂x[2,1]  ∂N∂x[1,1]]
        B2 = @SMatrix [∂N∂x[1,2]  0.0
                       0.0        ∂N∂x[2,2]
                       ∂N∂x[2,2]  ∂N∂x[1,2]]
        B3 = @SMatrix [∂N∂x[1,3]  0.0
                       0.0        ∂N∂x[2,3]
                       ∂N∂x[2,3]  ∂N∂x[1,3]]
        
        Bpart = B1*U1 + B2*U2 + B3*U3
        # -- Strain rate tensor
        ∂Ux∂x = Bpart[1]
        ∂Uz∂z = Bpart[2]
        ∂Ux∂z = mydot(view(∂N∂x,2,:),Ux)
        ∂Uz∂x = mydot(view(∂N∂x,1,:),Uz)
        # Velocity gradient
        L = @SMatrix [∂Ux∂x ∂Ux∂z
                      ∂Uz∂x ∂Uz∂z]
        # Update gradient of deformation
        F += Δt.*(L*F)
        particle_fields.Fxx[ipart] = F[1,1]
        particle_fields.Fxz[ipart] = F[1,2]
        particle_fields.Fzx[ipart] = F[2,1]
        particle_fields.Fzz[ipart] = F[2,2]
        
    end # end block loop
    
    return particle_fields
    
end # END OF ASSEMBLY FUNCTION

function volume_integral(V, EL2NOD, θ, r)
    ## MODEL PARAMETERS
    nip = 6
    nnodel = size(EL2NOD,1)
    nnodel = 6
    nel = size(EL2NOD,2)
    
    ## INTEGRATION POINTS & DERIVATIVES wrt LOCAL COORDINATES   
    ni, nn, nn3 = Val(nip), Val(nnodel), Val(3)
    N, ∇N,_ , w_ip = _get_SF(ni,nn)
    N3, b, dN3ds,c = _get_SF(ni,nn3)

    ## LOOP OVER ELEMENETS
    integral = zeros(nel)
    Threads.@threads for iel in 1:nel
        #= NOTE: For triangular elements with non-curved edges the Jacobian is
              the same for all integration points (i.e. calculated once
              before the integration loop). Further, linear 3-node shape are
              sufficient to calculate the Jacobian. =#

        # Polar coordinates of element nodes
        θ_el = @SVector [θ[i,iel] for i in 1:3]
        r_el = @SVector [r[i,iel] for i in 1:3]
        iel_coords = SMatrix{3,2}([θ_el r_el])
       
        # Jacobian n. 1 (p:=polar, l:=local): reference element --> current element
        J_pl = dN3ds*iel_coords
        detJ_pl = det(J_pl) # fast |-> unrolled
        
        # Jacobian ∂ξ∂θ to transform local (ξ, η) into global (θ,r) derivatives
        #     ∂ξ∂θ = [ R_31    -R_21
        #             -Th_31    Th_21] / detJa_PL
        # ∂ξ∂θ = @SMatrix [(r_el[3]-r_el[1])/detJ_pl -(r_el[2]-r_el[1])/detJ_pl
        #                 -(θ_el[3]-θ_el[1])/detJ_pl  (θ_el[2]-θ_el[1])/detJ_pl]
    
        idx = view(EL2NOD, 1:6, iel)
        V_el = SVector{6}(view(V, idx))

        ## INTEGRATION LOOP
        @inbounds for ip in 1:nip
            # Polar coordinates of the integration points
            r_ip = mydot(r_el,N3[ip])
            
            # Integration weight
            ω = r_ip*detJ_pl*w_ip[ip]
            
            # field at integration point
            V_ip = mydot(V_el,N[ip])
            
            # Integral reduction
            integral[iel] += V_ip*ω 
        end
        
    end # end block loop
    
    return sum(integral)
    
end # END OF ASSEMBLY FUNCTION

function volume_integral_cartesian(V, gr)
    ## MODEL PARAMETERS
    EL2NOD = gr.e2n
    x = gr.x[gr.e2n]
    z = gr.z[gr.e2n]
    nip = 6
    nnodel = size(EL2NOD,1)
    nnodel = 6
    nel = size(EL2NOD,2)
    
    ## INTEGRATION POINTS & DERIVATIVES wrt LOCAL COORDINATES   
    ni, nn, nn3 = Val(nip), Val(nnodel), Val(3)
    N, ∇N,_ , w_ip = _get_SF(ni,nn)
    _, b, dN3ds,c = _get_SF(ni,nn3)

    ## LOOP OVER ELEMENETS
    integral = zeros(nel)
    Threads.@threads for iel in 1:nel
        #= NOTE: For triangular elements with non-curved edges the Jacobian is
              the same for all integration points (i.e. calculated once
              before the integration loop). Further, linear 3-node shape are
              sufficient to calculate the Jacobian. =#

        # Polar coordinates of element nodes
        x_el = @SVector [x[i,iel] for i in 1:3]
        z_el = @SVector [z[i,iel] for i in 1:3]
        iel_coords = SMatrix{3,2}([x_el z_el])
       
        # Jacobian n. 1 (p:=polar, l:=local): reference element --> current element
        J_pl = dN3ds*iel_coords
        detJ_pl = det(J_pl) # fast |-> unrolled
        
        idx = view(EL2NOD, 1:6, iel)
        V_el = SVector{6}(view(V, idx))

        ## INTEGRATION LOOP
        @inbounds for ip in 1:nip                     
            # Integration weight
            ω = detJ_pl*w_ip[ip]
            
            # field at integration point
            V_ip = mydot(V_el, N[ip])
            # Integral reduction
            integral[iel] += V_ip*ω 
        end
        
    end # end block loop
    
    return sum(integral)
    
end # END OF ASSEMBLY FUNCTION
