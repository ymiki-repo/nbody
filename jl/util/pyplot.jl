module util_pyplot

using PyPlot
function config(; pkg="\\usepackage{amsmath}")
    rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")
    # embed fonts
    rcParams["ps.useafm"] = true
    rcParams["pdf.use14corefonts"] = true
    rcParams["text.usetex"] = true
    # use packages (physics.sty is missing on Wisteria/BDEC-01)
    rcParams["text.latex.preamble"] = pkg
    return nothing
end


using Parameters
@with_kw struct MyPlotType
    num::Int = 0
    type::Array{Any,1}
end

function call(base::MyPlotType; id::Integer=0)
    return base.type[begin+(id%base.num)]
end


using PyCall
@with_kw mutable struct Panel
    fig::Figure

    nx::Integer = 1
    ny::Integer = 1
    ax::Array{PyCall.PyObject,2} # axes

    fs::Int32 # font size
    ms::Float32 # marker size
    lw::Float32 # line width

    point::MyPlotType # point type
    line::MyPlotType # line style
    color::MyPlotType
    mono::MyPlotType # color (monochrome)
end

# chars: number of characters (for setting fontsize)
# dpi: dots per inch (for setting resolution)
# inch: size of panels in units of inch (A4 is 8.27 inch * 14.32 inch)
function set_Panel(; nx::Integer=1, ny::Integer=1, share_xaxis::Bool=true, share_yaxis::Bool=true, chars::Float32=24.0f0, dpi::Float32=300.0f0, inch::Float32=15.0f0, xscale::Float32=1.0f0, yscale::Float32=1.0f0)
    # set sizes (font size, marker size, line width, and tick length)
    fontsize = Int32(round(inch * 72 / chars)) # 72 pt = 1 inch
    markersize = inch
    linewidth = inch * 0.25f0
    ticklength = 6.0f0 * linewidth

    # configure axes
    xmin, xmax = 0.0f0, 1.0f0
    ymin, ymax = 0.0f0, 1.0f0
    xbin = (xmax - xmin) / Float32(nx)
    ybin = (ymax - ymin) / Float32(ny)
    xmargin, ymargin = 0.0f0, 0.0f0
    margin = 0.15f0
    if !share_yaxis
        xmin = 0.0f0
        xbin = 1.0f0 / Float32(nx)
        xmargin = xbin * margin
    end
    if !share_xaxis
        ymin = 0.0f0
        ybin = 1.0f0 / Float32(ny)
        ymargin = ybin * margin
    end

    # set default symbols
    pt = MyPlotType(num=5, type=["o", "s", "^", "D", "x"])
    ls = MyPlotType(num=5, type=["solid", (0, (1, 1)), (0, (5, 5)), (0, (5, 1, 1, 1)), (0, (5, 1, 1, 1, 1, 1, 1, 1))])
    cl = MyPlotType(num=10,
        # taken from Model Color Palette for Color Universal Design ver.4 (pages 7 and 2)
        # conversion using https://hogehoge.tk/tool/number.html
        type=[
            "#000000",# black
            "#ff4b00",# red
            "#005aff",# blue
            "#f6aa00",# orange
            "#03af7a",# green
            "#4dc4ff",# sky blue
            "#804000",# brown
            "#990099",# purple
            "#fff100",# yellow
            "#ff8082",# pink
        ]
    )
    mn = MyPlotType(num=4,
        # taken from Model Color Palette for Color Universal Design ver.4 (page 2)
        # conversion using https://hogehoge.tk/tool/number.html
        type=
        [
            "#000000",# black
            "#84919e",# dark gray
            "#c8c8cb",# light gray
            "#ffffff"# white
        ]
    )


    # construct structure
    panel = Panel(
        fig=PyPlot.figure(figsize=(inch * xscale * nx, inch * yscale * ny), dpi=dpi),
        nx=nx, ny=ny, ax=Array{PyCall.PyObject,2}(undef, (nx, ny)),
        fs=fontsize, ms=markersize, lw=linewidth,
        point=pt, line=ls, color=cl, mono=mn
    )

    # commit axes
    for jj in 1:ny
        yl = ymin + Float32(jj) * ybin + ymargin
        for ii in 1:nx
            xl = xmin + Float32(ii) * xbin + xmargin
            panel.ax[ii, jj] = panel.fig.add_axes((xl, yl, xbin - 2.0f0 * xmargin, ybin - 2.0f0 * ymargin))
        end
    end

    # configure axes
    for at in panel.ax
        for axis in ["top", "bottom", "left", "right"]
            at.spines[axis].set_linewidth(linewidth)
        end
        at.tick_params(axis="both", direction="in", bottom=true, top=true, left=true, right=true, labelsize=fontsize, length=ticklength, width=linewidth)
        at.tick_params(axis="x", pad=0.3f0 * fontsize)
        at.tick_params(axis="both", which="minor", direction="in", bottom=true, top=true, left=true, right=true, length=0.5f0 * ticklength, width=0.5f0 * linewidth)
        if share_xaxis
            at.tick_params(labelbottom=false)
        end
        if share_yaxis
            at.tick_params(labelleft=false)
        end
    end
    if share_xaxis
        for ii in 1:nx
            panel.ax[ii, begin].tick_params(labelbottom=true)
        end
    end
    if share_yaxis
        for jj in 1:ny
            panel.ax[begin, jj].tick_params(labelleft=true)
        end
    end

    return panel
end



function scale_axis(minimum::Real, maximum::Real; logPlt::Bool=true)
    blank_val = 0.2
    if logPlt
        width = log10(maximum / minimum)
        blank = width * blank_val * 0.5
        scale = 10.0^blank
        return (minimum / scale, maximum * scale)
    else
        width = maximum - minimum
        blank = width * blank_val * 0.5
        return (minimum - blank, maximum + blank)
    end
end

using Printf
function scientific(val::Real, pos)
    str = @sprintf "%.1e" val
    a, b = split(str, 'e')
    return string("\$", a, " \\times 10^{", parse(Int, b), "}\$")
end

end
