"""Instrument-aware guidance. Priors constrain language, never prove a defect."""
GUIDANCE = {
 'AIA_AR_sample': 'AIA is a solar imaging instrument with multiple EUV/UV/visible channels. Read a wavelength or product only if explicitly present. Intensity or display color alone does not measure magnetic polarity, temperature, or prove an event. Use the declared class vocabulary and coordinate convention.',
 'HMI_AR_sample': 'HMI has magnetogram, continuum and Doppler products. Do not assume which one is shown from the instrument name alone. If a magnetogram is explicitly established, its signal represents magnetic field, not brightness or temperature; polarity depends on the stated color scale. If the product or scale is absent, keep the physical interpretation unknown.',
 'CME_C2_sample': 'A coronagraph includes an occulting disk and field-of-view limits. The central dark area is not itself missing data. Position angle is circular: 0 and 360 degrees coincide. Interpret central angle, angular width, projection and timestamps according to the supplied definition; do not invent an orientation, cadence, speed or solar event time.',
 'EPB_seg_sample': 'This is a scientific plot/segmentation task, not a natural photograph. Axes, legends, data region and current annotation are separate. A visually dark or bright region is not automatically an EPB. Use declared image dimensions and coordinate units; do not silently rescale an annotation to conceal a dimension conflict.',
 'DMSP_SSUSI': 'SSUSI scientific images may use projected coordinates, color maps and masked/no-observation regions. Display color is not a class by itself. Compare current shapes with visible support using the provided projection/axis convention. Treat unavailable support or ambiguous projection as unknown, not a spurious event.',
 'ULF_sample': 'The observation is a numeric time series; the picture is only a waveform view. Use measured CSV time intervals, sample count, components and explicit units for parameter checks. An oscillation, apparent peak or flat curve alone does not establish a missing/spurious event. Event onset/offset requires an explicit event definition and coordinate/time-origin evidence.'
}
COMMON = ('Current observation and current provided annotation are the only inference inputs. '
          'The right panel is provided annotation, not a model prediction or defect truth. '
          'An empty annotation is not sufficient to claim a missing event; a provided annotation is not sufficient to claim a spurious event. '
          'Use only visible/measured evidence. Do not infer quality from filenames, package level or expected pollution rate. ')

def grounded_prompt(row):
    return COMMON + GUIDANCE[row['family']] + '\n' + row['prompt']
