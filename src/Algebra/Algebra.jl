# Out-of-place C = A*B
function gemm(A::AbstractMatrix, B::AbstractMatrix)
    m, n, p = size(A,1), size(B,2), size(A,2)

    C = Matrix{eltype(A)}(undef, m, n)

    @turbo for i in 1:m, k in 1:n
        aux = zero(eltype(A[1]))
        for j in 1:p
            aux += A[i,j]*B[j,k]
        end
        C[i,k] = aux
    end
    return C

end 

# threaded version of vectorized gemm
function gemmt(A::AbstractMatrix, B::AbstractMatrix)
    m, n, p = size(A,1), size(B,2), size(A,2)

    C = Matrix{eltype(A)}(undef, m, n)

    @tturbo for i in 1:m, k in 1:n
        aux = zero(eltype(A[1]))
        for j in 1:p
            aux += A[i,j]*B[j,k]
        end
        C[i,k] = aux
    end
    return C

end 

# In-place C = A*B
function gemm!(C::AbstractArray{T},A::AbstractArray{T},B::AbstractArray{T}) where T
    
    m, n, p = size(A,1), size(B,2), size(A,2)
    @turbo for i in 1:m, k in 1:n
        aux = zero(eltype(A[1]))
        for j in 1:p
            aux += A[i,j]*B[j,k]
        end
        C[i,k] = aux
    end
end 

function gemmt!(C::AbstractArray{T},A::AbstractArray{T},B::AbstractArray{T}) where T
    
    m, n, p = size(A,1), size(B,2), size(A,2)
    @tturbo for i in 1:m, k in 1:n
        aux = zero(eltype(A[1]))
        for j in 1:p
            aux += A[i,j]*B[j,k]
        end
        C[i,k] = aux
    end
end 