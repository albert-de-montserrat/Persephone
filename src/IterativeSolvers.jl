# PRECONDITIONED CONJUGATE GRADIENTS ================================================================
@inline  function StokesPcCG(U,P,KK,MM,GG,Rhs,ifree)
    
    rtol_Pat = 1e-8
    itmax_Pat = 200
    itmin_Pat = 10
    itnum = 0

    # fill!(U, 0.0)
    # fill!(P, 0.0)
    nU = length(Rhs) 
    z = fill(0.0, nU)
    
    # forces resulting from pressure gradients
    Rhs += GG*P
    # guess of the velocity field 
    U,F = _CholeskyFactorizationSolve(U, KK, Rhs, ifree) # return factorization F to speed up next direct solvers
    
    # ps, A_pardiso = _MKLfactorize(KK, Rhs, ifree)
    # _MKLsolve!(U, A_pardiso, ps, Rhs, ifree)

    # initial pressure residual vector
    GGtransp = GG'
    r = -GG'*U  
    rm0space!(r)
    Prms = mynorm(r) # norm of r_i
    tol = Prms * rtol_Pat
    # get preconditioner
    MM, pc = _preconditioner("jacobi", MM)
    # MM, pc = _preconditioner("lumped", MM)
    
    # Begin of Patera pressure iterations -------------------------------
    d = _precondition(pc, MM, r) # precondition residual
    # d = _precondition(pc, r) # precondition residual
    q = deepcopy(d)  # define FIRST search direction q

    for itPat = 1:itmax_Pat
        itnum +=1

        rm0space!(q)
        rd = mydot(r,d) # numerator for alpha
        
        #= Perform the S times q multiplication ======================    
            S cannot be calculated explicitly, since Kinv cannot be formed
            Sq = S * q = (G' * Kinv * G) * q
            Hence, the muliplicatipon is done in 3 steps:
            (1) y   = G*q
            (2) K z = y (TODO PARDISO direct solver)
            (3) Sq  = G'*z
        =============================================================#
        y = GG*q        
        _CholeskyWithFactorization!(z, F, y, ifree)
        # _MKLsolve!(z, A_pardiso, ps, y, ifree)
        Sq = GGtransp*z
        qSq = mydot(q,Sq) # denominator to calculate alpha
        rlast = copy(r) # needed for Polak-Ribiere version 1
        α = rd/qSq # steps size in direction q
        
        # Update solution and residual ------------------------------------ 
        _updatesolution!(P,U,r,q,z,Sq,α)

        rm0space!(r)
        # Check convergence -----------------------------------------------
        Prrms = mynorm(r)
        if Prrms < tol && itPat>=itmin_Pat
            println("\n", itnum," CG iterations\n")
            break
        end

        fill!(z,0.0)

        # Pressure preconditioning (approximate error of current P)
        d  = _precondition(pc, r) # precondition residual
        # d = _precondition(pc, MM, r) # precondition rsidual
        # Make new search direction q S-orthogonal to all previous q's
        β  = mydot(r-rlast,d)/rd # Polak-Ribiere version 1        
        q  = d + β*q # calculate NEW search direction
        
    end

    # _MKLrelease!(ps)

    return U,P

end # END PRECONDITIONED CONJUGATE GRADIENTS 

# Function takes out the mean of vector a
# (thereby removes the constant pressure mode, i.e. the nullspace)
@inline function rm0space!(a)
    a .-= mean(a);
end 

# PRECONDITIONERS ================================================================
@inline function _preconditioner(type, MM)
    if type == "diagonal"
        #=
            Inverse of the diagonal of the mass matrix scaled by viscosity
        =#
        MM, C = diagonalpreconditioner(MM)

    elseif type == "lumped"
        #=
            Inverse of lumped mass matrix scaled by viscosity
        =#        
        MM, C  = lumpedpreconditioner(MM)

    elseif type == "jacobi"
        #=
            Inverse of lumped mass matrix scaled by viscosity
        =#        
        MM, C  = lumpedpreconditioner(MM)
    end
    return MM, C
end # END OF PRECONDITIONERS

@inline function diagonalpreconditioner(MM)
    MM = MM .+ tril(MM,-1)'
    return MM, inv(diagm(MM))
end  

@inline function lumpedpreconditioner(MM)
    MM = MM .+ tril(MM,-1)'
    return MM, dropdims(sum(MM,dims=2), dims=2).^-1
end  

@inline function _updatesolution!(P,U,r,q,z,Sq,α)
    n1 = length(P)
    n2 = length(U)
    idx1 = 1:n1
    idx2 = (n1+1):n2
    @tturbo for i in idx1
        P[i] += α * q[i]
        U[i] += α * z[i]
        r[i] -= α * Sq[i]
    end
    @tturbo for i in idx2
        U[i] += α * z[i]
    end
end

# PRECONDITIONING: JACOBI ITERATIONS 
@inline function _precondition(pc::Array{Float64,1}, MM::SparseMatrixCSC{Float64,Int64} ,r::Array{Float64,1})
    weight = [0.2500, 0.4000, 0.7500]
    d = r .* pc
    dummy = Vector{Float64}(undef, length(d))
    @inbounds for is in 1:3
        # d = d .+ weight[is] * pc .* (r .- MM*d)
        d = d .+ weight[is] * pc .* (r .- mul!(dummy, MM, d))
    end
    d
end 

# PRECONDITIONING: lumped
@inline _precondition(pc::Vector,r::Vector) = r .* pc

function _CG!(T,KK,Rhs,ifree)
    # Load stiffness matrix and rhs
    A  = KK[ifree,ifree]
    b  = Vector{Float64}(undef,length(ifree))
    @turbo for  i ∈ 1:length(ifree) 
        b[i]   = Rhs[ifree[i]]        
    end
    # Diagonal preconditioner
    p = DiagonalPreconditioner(A)
    # p = AMGPreconditioner{SmoothedAggregation}(A)
    # Conjugate gradients
    x = cg(A, b, Pl=p)

    # Rebuild solution array
    c   = 0    
    @inbounds for i ∈ ifree        
        c   +=1
        T[i] = x[c]        
    end  
    
end


# end # END OF MODULE