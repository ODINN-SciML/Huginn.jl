export inn1

# Helper functions for the staggered grid

####  Non-allocating functions  ####

"""
    diff_x!(O, I, Δx)

Compute the finite difference of array `I` along the first dimension and store the result in array `O`. The difference is computed using the spacing `Δx`.

# Arguments
- `O`: Output array to store the finite differences.
- `I`: Input array from which finite differences are computed.
- `Δx`: Spacing between points in the first dimension.

# Notes
- The function uses `@views` to avoid copying data when slicing arrays.
- The operation is performed in-place, modifying the contents of `O`.
"""
diff_x!(O, I, Δx) = @views @. O = (I[begin+1:end,:] - I[1:end-1,:]) / Δx

"""
    diff_y!(O, I, Δy)

Compute the finite difference along the y-axis and store the result in `O`.

# Arguments
- `O`: Output array where the result will be stored.
- `I`: Input array from which the finite difference is computed.
- `Δy`: The spacing between points in the y-direction.

# Description
This function calculates the finite difference along the y-axis for the input array `I` and stores the result in the output array `O`. The calculation is performed using the formula:

    O = (I[:,begin+1:end] - I[:,1:end - 1]) / Δy

The `@views` macro is used to avoid copying data when slicing the array.
"""
diff_y!(O, I, Δy) = @views @. O = (I[:,begin+1:end] - I[:,1:end - 1]) / Δy

"""
    avg!(O, I)

Compute the average of adjacent elements in the input array `I` and store the result in the output array `O`.

# Arguments
- `O`: Output array where the averaged values will be stored.
- `I`: Input array from which the adjacent elements will be averaged.

# Details
This function uses the `@views` macro to avoid creating temporary arrays and the `@.` macro to broadcast the operations. The averaging is performed by taking the mean of each 2x2 block of elements in `I` and storing the result in the corresponding element in `O`.
"""
avg!(O, I) = @views @. O = (I[1:end-1,1:end-1] + I[2:end,1:end-1] + I[1:end-1,2:end] + I[2:end,2:end]) * 0.25

"""
    avg_x!(O, I)

Compute the average of adjacent elements along the first dimension of array `I` and store the result in array `O`.

# Arguments
- `O`: Output array where the averaged values will be stored.
- `I`: Input array from which adjacent elements will be averaged.
"""
avg_x!(O, I) = @views @. O = (I[1:end-1,:] + I[2:end,:]) * 0.5

"""
    avg_y!(O, I)

Compute the average of adjacent elements along the second dimension of array `I` and store the result in array `O`.

# Arguments
- `O`: Output array where the averaged values will be stored.
- `I`: Input array from which the adjacent elements will be averaged.
"""
avg_y!(O, I) = @views @. O = (I[:,1:end-1] + I[:,2:end]) * 0.5

####  Allocating functions  ####

"""
    avg(A::AbstractArray)

Compute the average of each 2x2 block in the input array `A`. The result is an array where each element is the average of the corresponding 2x2 block in `A`.

# Arguments
- `A::AbstractArray`: A 2D array of numerical values.

# Returns
- A 2D array of the same type as `A`, where each element is the average of a 2x2 block from `A`.
"""
@views avg(A) = 0.25 .* ( A[1:end-1,1:end-1] .+ A[2:end,1:end-1] .+ A[1:end-1,2:end] .+ A[2:end,2:end] )

"""
    avg_x(A::AbstractArray)

Compute the average of adjacent elements along the first dimension of the array `A`.

# Arguments
- `A::AbstractArray`: Input array.

# Returns
- An array of the same type as `A` with one less element along the first dimension, containing the averages of adjacent elements.
"""
@views avg_x(A) = 0.5 .* ( A[1:end-1,:] .+ A[2:end,:] )

"""
    avg_y(A::AbstractArray)

Compute the average of adjacent elements along the second dimension of the input array `A`.

# Arguments
- `A::AbstractArray`: An array of numeric values.

# Returns
- An array of the same type as `A` containing the averages of adjacent elements along the second dimension.
"""
@views avg_y(A) = 0.5 .* ( A[:,1:end-1] .+ A[:,2:end] )

"""
    diff_x(A::AbstractArray)

Compute the difference along the first dimension of the array `A`.

# Arguments
- `A::AbstractArray`: Input array.

# Returns
- An array of the same type as `A` containing the differences along the first dimension.
"""
@views diff_x(A) = (A[begin + 1:end, :] .- A[1:end - 1, :])

"""
    diff_y(A::AbstractArray)

Compute the difference between adjacent elements along the second dimension (columns) of the input array `A`.

# Arguments
- `A::AbstractArray`: An array of numeric values.

# Returns
- An array of the same type as `A` containing the differences between adjacent elements along the second dimension.
"""
@views diff_y(A) = (A[:, begin + 1:end] .- A[:, 1:end - 1])

"""
    inn(A::AbstractArray)

Extracts the inner part of a 2D array `A`, excluding the first and last rows and columns.

# Arguments
- `A::AbstractArray`: A 2D array from which the inner part will be extracted.

# Returns
- A subarray of `A` containing all elements except the first and last rows and columns.
"""
@views inn(A) = A[2:end-1,2:end-1]

"""
    inn1(A::AbstractArray)

Returns a view of the input array `A` excluding the last row and the last column.

# Arguments
- `A::AbstractArray`: The input array from which a subarray view is created.

# Returns
- A view of the input array `A` that includes all elements except the last row and the last column.
"""
@views inn1(A) = A[1:end-1,1:end-1]


### Enzyme scalar implementation of util functions

function avg_enzyme!(R, A)
    @inbounds @simd for i in eachindex(R)
        R[i] = 0
    end

    @inbounds for i in axes(R, 1)
        @inbounds @simd for j in axes(R,2)
            R[i,j] += 0.25 * (A[i,j] + A[i+1,j] + A[i,j+1] + A[i+1,j+1])
        end
    end
end

function avg_x_enzyme!(R, A)
    @inbounds @simd for i in eachindex(R)
        R[i] = 0
    end

    @inbounds for i in axes(R, 1)
        @inbounds @simd for j in axes(R,2)
            R[i,j] += 0.50 * (A[i,j] + A[i+1,j])
        end
    end
end

function avg_y_enzyme!(R, A)
    @inbounds @simd for i in eachindex(R)
        R[i] = 0
    end

    @inbounds for i in axes(R, 1)
        @inbounds @simd for j in axes(R,2)
            R[i,j] += 0.50 * (A[i,j] + A[i,j+1])
        end
    end
end

function diff_x_enzyme!(R, A)
    @inbounds @simd for i in eachindex(R)
        R[i] = 0
    end

    @inbounds for i in axes(R, 1)
        @inbounds @simd for j in axes(R,2)
            R[i,j] += A[i+1,j] - A[i,j]
        end
    end
end

function diff_y_enzyme!(R, A)
    @inbounds @simd for i in eachindex(R)
        R[i] = 0
    end

    @inbounds for i in axes(R, 1)
        @inbounds @simd for j in axes(R,2)
            R[i,j] += A[i,j+1] - A[i,j]
        end
    end
end