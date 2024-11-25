#using Plots
using Images

function convert_to_gray_image(M) 
    minval, maxval = minimum(M), maximum(M)
    M2 = clamp.((M .- minval) ./ max(1e-6, maxval-minval), 0.0, 1.0)
    return convert(Matrix{Gray{N0f8}}, M2)
end

# Note: each filter is a *row* in W
function render_filters(W, nx, ny, dx, dy; border = 1, individual_scaling = true, center_gray = true)
    nfilters = min(size(W,1), nx*ny); dim = size(W,2)
    @assert dx*dy == dim
    W0 = W[1:nfilters, :]

    minval, maxval = minimum(W0), maximum(W0)
    if individual_scaling
        minval = minimum(W0, dims=2)
        maxval = maximum(W0, dims=2)
    end

    if center_gray
        minval = min.(minval, -maxval)
        maxval = max.(maxval, -minval)
    end

    W1 = clamp.((W0 .- minval) ./ max.(1e-6, maxval-minval), 0.0, 1.0)
    W2 = reshape(W1, nfilters, dx, dy)

    W3 = zeros(Float32, (nx*dx + border*(nx-1), ny*dy + border*(ny-1)))
    for x = 0:nx-1, y = 0:ny-1
        f = 1 + x + nx*y
        x0 = 1 + x*(dx + border); y0 = 1 + y*(dy + border)
        if f <= nfilters
            W3[x0:(x0+dx-1), y0:(y0+dy-1)] = W2[f, :, :]
        end
    end
    #@show size(W3)
    return convert(Matrix{Gray{N0f8}}, W3')
end

function render_patches(W, nx, ny, dx, dy; border = 1, invert = false)
    nfilters = min(size(W,1), nx*ny); dim = size(W,2)
    @assert dx*dy == dim
    W0 = W[1:nfilters, :]

    W1 = clamp.(W0 , 0, 1)
    W2 = reshape(W1, nfilters, dx, dy)

    W3 = zeros(Float32, (nx*dx + border*(nx-1), ny*dy + border*(ny-1)))
    for x = 0:nx-1, y = 0:ny-1
        f = 1 + x + nx*y
        x0 = 1 + x*(dx + border); y0 = 1 + y*(dy + border)
        if f <= nfilters
            W3[x0:(x0+dx-1), y0:(y0+dy-1)] = invert ? (1 .- W2[f, :, :]) : W2[f, :, :]
        end
    end
    #@show size(W3)
    return convert(Matrix{Gray{N0f8}}, W3')
end

#function display_filters(W, nx, ny, dx, dy)
#    plot(render_filters(W, nx, ny, dx, dy))
#end
