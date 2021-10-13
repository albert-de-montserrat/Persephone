```
Annealing ratio ∈ [0, 1] is linear within upper 
and lower bound strain rates (ε1, ε2) defined by the user
```
struct Annealing{T}
    ε1::T
    ε2::T
end

```
annealing_ratio(εII, annealing::Annealing) -> s
```
function annealing_ratio(εII, annealing)
    s = similar(εII)
    slope = 1/(annealing.ε2 - annealing.ε1)
    @batch for i in eachindex(s)
        s[i] = applybounds(εII[i]*slope, 0, 1) # s ∈ [0, 1]
    end
    s
end


```
annealing_ratio!(s, εII, annealing::Annealing) 
```
function annealing_ratio!(s, εII, annealing)
    slope = 1/(annealing.ε2 - annealing.ε1)
    @batch for i in eachindex(s)
        s[i] = applybounds(εII[i]*slope, 0, 1) # s ∈ [0, 1]
    end
end

```
unstretch_axes(a1, a2, s) -> a1, a2

Unstrecth the FSE semi-axes by a factor of s.
The area of the ellipse must remain the same
after unstretching
```
function unstretch_axes(a1, a2, s)
    # a2, a1 are the old semiaxes
    # s is the unstretching factor
    r0 = a2/a1 # old ratio 
    r = r0*(1+s) # new aspect ratio
    A = π*a2*a1 # Area of the FSE 
    # find new axes such that new aspect 
    # ratio is r and A remains untouched
    a1n = √(A/(π*r))
    a2n = r*a1n
    return a1n, a2n
end

```
recover_FSE(F, l0, λ11, λ12, λ21, λ22, a1, a2) -> F

Recover the new F tensor after unstretching the FSE semi-axes.
We know that F = LR, where L is the left-stretch tensor and R⁻¹=Rᵀ 
is the orthogonal rotation tensor. First we recover the new L using
the old eigenvalues (we do not want to rotate the FSE, only unstretch 
it) and new eigenvalues:
    L = PDP⁻¹
where P = [λ₁ λ₃ λ₃] and D = I⋅[a₁ a₃ a₃]ᵀ. R is obtained using the 
old F and L0 tensors:
    R = F\L0
and finally we obtain the new deformation gradient tensor:
    Fn = LR
```
function recover_FSE(F, L0, λ11, λ12, λ21, λ22, a1, a2)
    P = @SMatrix [
        λ11 λ21
        λ12 λ22
    ]
    D = @SMatrix [
        a1*a1  0
        0      a2*a2
    ]
    # recover left-stretch tensor 
    L = P*D\P
    # rotation tensor R = F\(F*Fᵀ) = F\l0
    R = F\L0
    Fn = L*R
end