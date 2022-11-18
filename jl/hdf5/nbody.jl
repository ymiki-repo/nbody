module hdf5_nbody

using Parameters

@with_kw mutable struct Conservatives
    # num::Integer = 0
    time::Array{Real,1}
    Etot::Array{Real,1}
    Ekin::Array{Real,1}
    Epot::Array{Real,1}
    err_Etot::Array{Real,1}
    err_px::Array{Real,1}
    err_py::Array{Real,1}
    err_pz::Array{Real,1}
    err_Lx::Array{Real,1}
    err_Ly::Array{Real,1}
    err_Lz::Array{Real,1}
end

@with_kw mutable struct Particles
    pot::Array{Real,1}
    pos::Array{Real,2}
    vel::Array{Real,2}
    time::Real
end


using HDF5
using FilePaths

function read_conservatives(files::Array{String,1})
    num = length(files)
    csv = Conservatives(
        time=Array{Real,1}(undef, num),
        Etot=Array{Real,1}(undef, num),
        Ekin=Array{Real,1}(undef, num),
        Epot=Array{Real,1}(undef, num),
        err_Etot=Array{Real,1}(undef, num),
        err_px=Array{Real,1}(undef, num),
        err_py=Array{Real,1}(undef, num),
        err_pz=Array{Real,1}(undef, num),
        err_Lx=Array{Real,1}(undef, num),
        err_Ly=Array{Real,1}(undef, num),
        err_Lz=Array{Real,1}(undef, num)
    )

    # read snapshots
    for file in files
        idx = parse(Int, split(filename(Path(file)), "_snp")[begin+1]) + 1# array is Fortran manner

        HDF5.h5open(file, "r") do loc
            csv.time[idx] = HDF5.read_attribute(loc, "time")[begin]
            grp = HDF5.open_group(loc, "conservative/current")
            csv.Etot[idx] = HDF5.read_attribute(grp, "E_tot")[begin]
            csv.Ekin[idx] = HDF5.read_attribute(grp, "E_kin")[begin]
            csv.Epot[idx] = HDF5.read_attribute(grp, "E_pot")[begin]
            HDF5.close(grp)
            grp = HDF5.open_group(loc, "conservative/latest error")
            csv.err_Etot[idx] = HDF5.read_attribute(grp, "E_tot")[begin]
            csv.err_px[idx] = HDF5.read_attribute(grp, "px")[begin]
            csv.err_py[idx] = HDF5.read_attribute(grp, "py")[begin]
            csv.err_pz[idx] = HDF5.read_attribute(grp, "pz")[begin]
            csv.err_Lx[idx] = HDF5.read_attribute(grp, "Lx")[begin]
            csv.err_Ly[idx] = HDF5.read_attribute(grp, "Ly")[begin]
            csv.err_Lz[idx] = HDF5.read_attribute(grp, "Lz")[begin]
            HDF5.close(grp)
        end
    end

    return csv
end

function read_particle(file::String, Ntot::Integer)
    body = Particles(
        pot=Array{Real,1}(undef, Ntot),
        pos=Array{Real,2}(undef, (3, Ntot)),
        vel=Array{Real,2}(undef, (3, Ntot)),
        time=0.0
    )

    # read snapshots
    HDF5.h5open(file, "r") do loc
        body.pot = vec(HDF5.read_dataset(loc, "particle/pot"))
        body.pos = reshape(vec(HDF5.read_dataset(loc, "particle/pos")), (3, Ntot))
        body.vel = reshape(vec(HDF5.read_dataset(loc, "particle/vel")), (3, Ntot))

        # read current time
        body.time = HDF5.read_attribute(loc, "time")[begin]
    end

    return body
end


end
