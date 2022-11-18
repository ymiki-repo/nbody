using ArgParse
function parse_cmd()
    cfg = ArgParseSettings()
    @add_arg_table cfg begin
        "--target", "-t"
        help = "filename of the target files"
        arg_type = String
        default = "collapse"
        "--pdf", "-p"
        help = "generate figure in PDF format"
        action = :store_true
    end
    return parse_args(cfg)
end

using Parameters
@with_kw mutable struct Conservatives
    param::Array{Real,1}
    err_Etot::Array{Real,1}
    worst_err_Etot::Array{Real,1}
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
    )
    return results, have_eta
end


using PyPlot
include("../util/pyplot.jl")


function main()
    # read options
    argv = parse_cmd()
    output_pdf = argv["pdf"]

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
    fig_ene.ax[begin].set_ylabel(L"$\abs{E(t) / E(t = 0) - 1}$", fontsize=fig_ene.fs)
    fig_ene.ax[begin].loglog()
    fig_ene.ax[begin].grid()
    handles, labels = fig_ene.ax[begin].get_legend_handles_labels()
    fig_ene.ax[begin].legend(handles[end:-1:begin], labels[end:-1:begin], numpoints=1, handlelength=2.0, loc="best", fontsize=fig_ene.fs)

    # save figures
    fig_ene.fig.savefig(string("fig/", series, "_scl_ene", ".png"), format="png", dpi=100, bbox_inches="tight")
    if output_pdf
        fig_ene.fig.savefig(string("fig/", series, "_scl_ene", ".pdf"), format="pdf", bbox_inches="tight")
    end

    fig_ene = nothing
    PyPlot.close("all")
    return nothing
end


main()
