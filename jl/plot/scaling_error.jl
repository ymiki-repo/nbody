using ArgParse
function parse_cmd()
    cfg = ArgParseSettings()
    @add_arg_table cfg begin
        "--target", "-t"
        help = "filename of the target files"
        arg_type = String
        default = "collapse"
        "--pdf"
        help = "generate figure in PDF format"
        action = :store_true
        "--png"
        help = "generate figure in PNG format"
        action = :store_true
        "--svg"
        help = "generate figure in SVG format"
        action = :store_true
    end
    return parse_args(cfg)
end

using Parameters
@with_kw mutable struct Conservatives
    param::Array{Real,1}
    err_Etot::Array{Real,1}
    worst_err_Etot::Array{Real,1}
    elapse_time::Array{Real,1}
    time_per_step::Array{Real,1}
    FP_L::Array{Integer,1}
    FP_M::Array{Integer,1}
end

using DataFrames
using CSV
function read_csv(file)
    csv = DataFrame(CSV.File(file))
    have_eta = "eta" in names(csv)
    results = Conservatives(
        param=csv[!, have_eta ? :eta : :dt],
        err_Etot=csv[!, :energy_error_final],
        worst_err_Etot=csv[!, :energy_error_worst],
        elapse_time=csv[!, :"time[s]"],
        time_per_step=csv[!, :"time_per_step[s]"],
        FP_L=csv[!, :"FP_L"],
        FP_M=csv[!, :"FP_M"]
    )
    return results, have_eta
end


using PyPlot
include("../util/pyplot.jl")


function main()
    # read options
    argv = parse_cmd()
    output_pdf = argv["pdf"]
    output_png = argv["png"]
    output_svg = argv["svg"]

    # find the latest series of simulation results
    series = argv["target"]

    # read CSV file
    dat, have_eta = read_csv(string("log/", series, "_run.csv"))
    xlabel = have_eta ? L"$\eta$" : L"$\varDelta t$"

    # initialize matplotlib
    util_pyplot.config()

    # show error scaling of the energy conservation
    fig_ene = util_pyplot.set_Panel()
    fig_ene.ax[begin].plot(dat.param, abs.(dat.worst_err_Etot), util_pyplot.call(fig_ene.point, id=1), color=util_pyplot.call(fig_ene.color, id=0), markersize=fig_ene.ms, label=L"$t = t_\mathrm{worst}$")
    fig_ene.ax[begin].plot(dat.param, abs.(dat.err_Etot), util_pyplot.call(fig_ene.point, id=0), color=util_pyplot.call(fig_ene.color, id=1), markersize=fig_ene.ms, label=L"$t = t_\mathrm{final}$")
    fig_ene.ax[begin].set_xlabel(xlabel, fontsize=fig_ene.fs)
    fig_ene.ax[begin].set_ylabel(L"$\left|E(t) / E(t = 0) - 1\right|$", fontsize=fig_ene.fs)
    fig_ene.ax[begin].loglog()
    fig_ene.ax[begin].grid()
    handles, labels = fig_ene.ax[begin].get_legend_handles_labels()
    fig_ene.ax[begin].legend(handles[end:-1:begin], labels[end:-1:begin], numpoints=1, handlelength=2.0, loc="best", fontsize=fig_ene.fs)

    # show error scaling of the energy conservation (with number of bits for floating-point numbers)
    fig_ene_fp = util_pyplot.set_Panel()
    fig_ene_fp.ax[begin].plot(dat.param[(dat.FP_L.==32).&&(dat.FP_M.==32)], abs.(dat.err_Etot[(dat.FP_L.==32).&&(dat.FP_M.==32)]), util_pyplot.call(fig_ene_fp.point, id=0), color=util_pyplot.call(fig_ene_fp.color, id=0), markersize=fig_ene_fp.ms, linestyle=util_pyplot.call(fig_ene_fp.line, id=0), linewidth=fig_ene_fp.lw, label="Low: FP32, Mid: FP32")
    fig_ene_fp.ax[begin].plot(dat.param[(dat.FP_L.==32).&&(dat.FP_M.==64)], abs.(dat.err_Etot[(dat.FP_L.==32).&&(dat.FP_M.==64)]), util_pyplot.call(fig_ene_fp.point, id=1), color=util_pyplot.call(fig_ene_fp.color, id=1), markersize=fig_ene_fp.ms, linestyle=util_pyplot.call(fig_ene_fp.line, id=1), linewidth=fig_ene_fp.lw, label="Low: FP32, Mid: FP64")
    fig_ene_fp.ax[begin].plot(dat.param[(dat.FP_L.==64).&&(dat.FP_M.==64)], abs.(dat.err_Etot[(dat.FP_L.==64).&&(dat.FP_M.==64)]), util_pyplot.call(fig_ene_fp.point, id=2), color=util_pyplot.call(fig_ene_fp.color, id=2), markersize=fig_ene_fp.ms, linestyle=util_pyplot.call(fig_ene_fp.line, id=2), linewidth=fig_ene_fp.lw, label="Low: FP64, Mid: FP64")
    fig_ene_fp.ax[begin].set_xlabel(xlabel, fontsize=fig_ene_fp.fs)
    fig_ene_fp.ax[begin].set_ylabel(L"$\left|E(t) / E(t = 0) - 1\right|$", fontsize=fig_ene_fp.fs)
    fig_ene_fp.ax[begin].loglog()
    fig_ene_fp.ax[begin].grid()
    handles, labels = fig_ene_fp.ax[begin].get_legend_handles_labels()
    fig_ene_fp.ax[begin].legend(handles[end:-1:begin], labels[end:-1:begin], numpoints=1, handlelength=2.0, loc="best", fontsize=fig_ene_fp.fs)

    # show execution time per step as a function of conservation error (with number of bits for floating-point numbers)
    fig_elapse_err_fp = util_pyplot.set_Panel()
    fig_elapse_err_fp.ax[begin].plot(abs.(dat.err_Etot[(dat.FP_L.==32).&&(dat.FP_M.==32)]), dat.elapse_time[(dat.FP_L.==32).&&(dat.FP_M.==32)], util_pyplot.call(fig_elapse_err_fp.point, id=0), color=util_pyplot.call(fig_elapse_err_fp.color, id=0), markersize=fig_elapse_err_fp.ms, linestyle=util_pyplot.call(fig_elapse_err_fp.line, id=0), linewidth=fig_elapse_err_fp.lw, label="Low: FP32, Mid: FP32")
    fig_elapse_err_fp.ax[begin].plot(abs.(dat.err_Etot[(dat.FP_L.==32).&&(dat.FP_M.==64)]), dat.elapse_time[(dat.FP_L.==32).&&(dat.FP_M.==64)], util_pyplot.call(fig_elapse_err_fp.point, id=1), color=util_pyplot.call(fig_elapse_err_fp.color, id=1), markersize=fig_elapse_err_fp.ms, linestyle=util_pyplot.call(fig_elapse_err_fp.line, id=1), linewidth=fig_elapse_err_fp.lw, label="Low: FP32, Mid: FP64")
    fig_elapse_err_fp.ax[begin].plot(abs.(dat.err_Etot[(dat.FP_L.==64).&&(dat.FP_M.==64)]), dat.elapse_time[(dat.FP_L.==64).&&(dat.FP_M.==64)], util_pyplot.call(fig_elapse_err_fp.point, id=2), color=util_pyplot.call(fig_elapse_err_fp.color, id=2), markersize=fig_elapse_err_fp.ms, linestyle=util_pyplot.call(fig_elapse_err_fp.line, id=2), linewidth=fig_elapse_err_fp.lw, label="Low: FP64, Mid: FP64")
    fig_elapse_err_fp.ax[begin].set_xlabel(L"$\left|E(t) / E(t = 0) - 1\right|$", fontsize=fig_elapse_err_fp.fs)
    fig_elapse_err_fp.ax[begin].set_ylabel(string("Time to solution", L"~$\left[\mathrm{s}\right]$"), fontsize=fig_elapse_err_fp.fs)
    fig_elapse_err_fp.ax[begin].loglog()
    fig_elapse_err_fp.ax[begin].grid()
    handles, labels = fig_elapse_err_fp.ax[begin].get_legend_handles_labels()
    fig_elapse_err_fp.ax[begin].legend(handles[end:-1:begin], labels[end:-1:begin], numpoints=1, handlelength=2.0, loc="best", fontsize=fig_elapse_err_fp.fs)

    # save figures
    if output_png
        fig_ene.fig.savefig(string("fig/", series, "_scl_err_ene", ".png"), format="png", dpi=100, bbox_inches="tight")
        fig_ene_fp.fig.savefig(string("fig/", series, "_scl_err_ene_fp", ".png"), format="png", dpi=100, bbox_inches="tight")
        fig_elapse_err_fp.fig.savefig(string("fig/", series, "_elapse_err_ene_fp", ".png"), format="png", dpi=100, bbox_inches="tight")
    end
    if output_pdf
        fig_ene.fig.savefig(string("fig/", series, "_scl_err_ene", ".pdf"), format="pdf", bbox_inches="tight")
        fig_ene_fp.fig.savefig(string("fig/", series, "_scl_err_ene_fp", ".pdf"), format="pdf", bbox_inches="tight")
        fig_elapse_err_fp.fig.savefig(string("fig/", series, "_elapse_err_ene_fp", ".pdf"), format="pdf", bbox_inches="tight")
    end
    if output_svg
        fig_ene.fig.savefig(string("fig/", series, "_scl_err_ene", ".svg"), format="svg", dpi=100, bbox_inches="tight")
        fig_ene_fp.fig.savefig(string("fig/", series, "_scl_err_ene_fp", ".svg"), format="svg", dpi=100, bbox_inches="tight")
        fig_elapse_err_fp.fig.savefig(string("fig/", series, "_elapse_err_ene_fp", ".svg"), format="svg", dpi=100, bbox_inches="tight")
    end

    fig_ene = nothing
    fig_ene_fp = nothing
    fig_elapse_err_fp = nothing
    PyPlot.close("all")
    return nothing
end


main()
