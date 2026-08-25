//
// ESP32-S3 PIE column-pass IDCT butterfly for JPEG decompression.
// Forked/authored for the S3-nextgen jpeg project, building on the
// dequant/color-convert SIMD kernels and the abandoned column-pass sketch
// (s3_simd_idct.S_FUTURE) originally started by Larry Bank (BitBank Software).
//
// SCOPE: this does ONLY the AA&N column-pass butterfly (the "even part" /
// "odd part" math that combines rows 0/2/4/6 and 1/3/5/7 into 8 new rows),
// always taking the "full calculation" branch. It does NOT vectorize the
// row pass or the final clamp-to-pixel stage — PIE has no shuffle/transpose/
// gather/horizontal-add instructions (confirmed against Espressif's own PIE
// writeups), so the row pass (which needs to combine elements *within* a
// single row/register) cannot be done the same way the column pass can
// (which combines elements across *different* registers, lane-by-lane).
// Row pass and output clamp stay scalar in JPEGIDCT, unchanged.
//
// This also always takes the "rows 4-7 populated" full-calculation path,
// skipping the scalar/SSE/NEON code's optimization for MCUs whose bottom
// half is entirely zero. That shortcut is safe to skip (0 * quant = 0,
// still numerically correct) - it just means blocks that could have taken
// a faster path don't get it in the SIMD version. Traded deliberately for
// a much smaller/safer register schedule; see conversation notes.
//
// Call as: void s3_idct_columns(int16_t *pMCU, const int16_t *pConst);
// pMCU must already be dequantized in place (call s3_dequant() first).
// pConst layout: 4 blocks of 8x replicated int16, 16 bytes each, in order
//   [0]  K1414   = round(1.414213562 * 2^14) = 23170   (SAR 14)
//   [16] K1847   = round(1.847759065 * 2^14) = 30274   (SAR 14)
//   [32] KNEG2613= round(-2.613125930 * 2^13) = -21407  (SAR 13, used as a
//                  single multiply with NO further scaling - this constant
//                  overflows a signed 16-bit value at 14 fractional bits,
//                  which is why it alone uses 13; unlike the SSE/NEON
//                  reference's own encoding of this constant, ours needs no
//                  post-multiply doubling - see scratchpad/idct_columns_
//                  validate.c, an earlier draft copied the SSE "double
//                  after" step needlessly and it produced ~65000-magnitude
//                  garbage until removed)
//   [48] K10824  = round(1.082392200 * 2^14) = 17735   (SAR 14)
// See scratchpad/idct_const_gen.py for how these were derived and for the
// bit-exact C reference this assembly is scheduled against.
//
#ifdef ARDUINO_ESP32S3_DEV

#include "dsps_fft2r_platform.h"
#if (dsps_fft2r_sc16_aes3_enabled == 1)
	.text
	.align 4
//                             A2        A3
// Call as void s3_idct_columns(int16_t *pMCU, const int16_t *pConst);
	.global s3_idct_columns
	.type   s3_idct_columns,@function

s3_idct_columns:
	entry	a1,64          # 64-byte frame: spill area for the 4 even-part
	                       # results (tmp0..tmp3), 16 bytes each.

	# ---------------- EVEN PART (rows 0, 2, 4, 6) ----------------
	# All addressing here uses `addi` to compute an explicit byte offset
	# from the (untouched) base pointer in a2, rather than chaining
	# ee.vld.128.ip's auto-increment across non-sequential rows. That
	# auto-increment trick is only safe for strictly-ascending access
	# (see s3_dequant.S); rows are visited out of order here (0,4,2,6),
	# so each load gets its own explicitly-computed address instead.
	addi	a5,a2,0        # &row0
	ee.vld.128.ip	q0,a5,0    # q0 = row0
	addi	a5,a2,64       # &row4 (row*16 bytes; row4 = 4*16)
	ee.vld.128.ip	q1,a5,0    # q1 = row4
	ee.vadds.s16	q2,q0,q1   # q2 = tmp10 = row0+row4
	ee.vsubs.s16	q3,q0,q1   # q3 = tmp11 = row0-row4

	addi	a5,a2,32       # &row2
	ee.vld.128.ip	q0,a5,0    # q0 = row2 (row0 no longer needed)
	addi	a5,a2,96       # &row6
	ee.vld.128.ip	q1,a5,0    # q1 = row6
	ee.vadds.s16	q4,q0,q1   # q4 = tmp13 = row2+row6
	ee.vsubs.s16	q0,q0,q1   # q0 = row2-row6 (row2,row6 dead after this)

	movi.n	a6,14
	wsr.sar	a6             # SAR = 14 fractional bits for the constants below
	addi	a5,a3,0        # &K1414
	ee.vld.128.ip	q1,a5,0    # q1 = K1414
	ee.vmul.s16	q1,q0,q1   # q1 = (row2-row6) * K1414   [>>14]
	ee.vsubs.s16	q1,q1,q4   # q1 = tmp12 = that - tmp13

	ee.vadds.s16	q0,q2,q4   # q0 = tmp0 (FINAL even)  = tmp10+tmp13
	ee.vsubs.s16	q4,q2,q4   # q4 = tmp3 (FINAL even)  = tmp10-tmp13
	ee.vadds.s16	q2,q3,q1   # q2 = tmp1 (FINAL even)  = tmp11+tmp12
	ee.vsubs.s16	q3,q3,q1   # q3 = tmp2 (FINAL even)  = tmp11-tmp12

	# Spill all 4 even-part results - the odd part below needs the full
	# 8-register file as working space.
	mov.n	a9,a1
	ee.vst.128.ip	q0,a9,16   # spill tmp0 @ +0
	ee.vst.128.ip	q2,a9,16   # spill tmp1 @ +16
	ee.vst.128.ip	q3,a9,16   # spill tmp2 @ +32
	ee.vst.128.ip	q4,a9,16   # spill tmp3 @ +48

	# ---------------- ODD PART (rows 1, 3, 5, 7) ----------------
	addi	a5,a2,48       # &row3
	ee.vld.128.ip	q0,a5,0    # q0 = row3 ("tmp5" raw)
	addi	a5,a2,80       # &row5
	ee.vld.128.ip	q1,a5,0    # q1 = row5 ("tmp6" raw)
	ee.vadds.s16	q2,q1,q0   # q2 = z13 = tmp6+tmp5
	ee.vsubs.s16	q3,q1,q0   # q3 = z10 = tmp6-tmp5

	addi	a5,a2,16       # &row1
	ee.vld.128.ip	q0,a5,0    # q0 = row1 ("tmp4" raw)
	addi	a5,a2,112      # &row7
	ee.vld.128.ip	q1,a5,0    # q1 = row7 ("tmp7" raw)
	ee.vadds.s16	q4,q0,q1   # q4 = z11 = tmp4+tmp7
	ee.vsubs.s16	q5,q0,q1   # q5 = z12 = tmp4-tmp7

	# live: q2=z13 q3=z10 q4=z11 q5=z12 | free: q0 q1 q6 q7
	ee.vadds.s16	q6,q4,q2   # q6 = tmp7 (FINAL odd) = z11+z13  [kept live to the end]
	ee.vsubs.s16	q0,q4,q2   # q0 = z11-z13                     (q4,q2 now dead)

	addi	a5,a3,0        # &K1414 (SAR still 14 from the even part)
	ee.vld.128.ip	q1,a5,0
	ee.vmul.s16	q1,q0,q1   # q1 = tmp11 = (z11-z13) * K1414   [>>14]

	# live: q1=tmp11 q3=z10 q5=z12 q6=tmp7final | free: q0 q2 q4 q7
	ee.vadds.s16	q0,q3,q5   # q0 = z10+z12
	addi	a5,a3,16       # &K1847
	ee.vld.128.ip	q2,a5,0
	ee.vmul.s16	q0,q0,q2   # q0 = z5 = (z10+z12) * K1847      [>>14]
	                       # z5 has TWO later uses (tmp12 and tmp10) - kept
	                       # live across both; this is the exact spot the
	                       # naive schedule loses it if you're not careful.

	addi	a5,a3,32       # &KNEG2613 (already negative, scaled for SAR13)
	ee.vld.128.ip	q2,a5,0
	movi.n	a6,13
	wsr.sar	a6             # this constant needs 13 fractional bits to avoid
	                       # signed-16-bit overflow; doubled below to
	                       # compensate, same trick as the SSE/NEON reference.
	ee.vmul.s16	q4,q3,q2   # q4 = z10 * KNEG2613               [>>13]  (z10/q3 now dead)
	ee.vadds.s16	q4,q4,q0   # q4 = tmp12 = that + z5           (z5/q0 still needed once more)

	# live: q0=z5 q1=tmp11 q4=tmp12 q5=z12 q6=tmp7final | free: q2 q3 q7
	ee.vsubs.s16	q2,q4,q6   # q2 = tmp6 (FINAL odd) = tmp12-tmp7   (tmp12/q4 dead)
	ee.vsubs.s16	q3,q1,q2   # q3 = tmp5 (FINAL odd) = tmp11-tmp6   (tmp11/q1 dead)

	# live: q0=z5 q2=tmp6final q3=tmp5final q5=z12 q6=tmp7final | free: q1 q4 q7
	movi.n	a6,14
	wsr.sar	a6
	addi	a5,a3,48       # &K10824
	ee.vld.128.ip	q1,a5,0
	ee.vmul.s16	q1,q5,q1   # q1 = z12 * K10824                 [>>14]  (z12/q5 dead)
	ee.vsubs.s16	q1,q1,q0   # q1 = tmp10 = that - z5           (z5/q0 dead - last use)
	ee.vadds.s16	q1,q1,q3   # q1 = tmp4 (FINAL odd) = tmp10+tmp5
	                       # (reads q3 without destroying it - tmp5 is still
	                       # needed for the row2/row5 combine below)

	# Odd part done. Final state: q1=tmp4 q2=tmp6 q3=tmp5 q6=tmp7
	# Free: q0 q4 q5 q7  -  exactly 4 free registers for the 4 spilled
	# even-part results. (This is the number that has to come out exactly
	# right for the whole schedule to fit in 8 registers; it does.)

	# ---------------- Reload 3 of the 4 even results ----------------
	# Only 3, not 4: the final combine needs one spare register to hold
	# a sum/diff before storing it (see comments below), and reloading
	# all 4 immediately would leave zero registers free for that. tmp3
	# is reloaded later, right when register q1 (tmp4) frees up for it.
	mov.n	a9,a1
	ee.vld.128.ip	q0,a9,16   # q0 = tmp0
	ee.vld.128.ip	q4,a9,16   # q4 = tmp1
	ee.vld.128.ip	q5,a9,16   # q5 = tmp2
	                       # a9 now points at the spilled tmp3, saved for later.

	# ---------------- Final combine + store ----------------
	# Each pair (tmpN, tmp(7-N)) produces two output rows: sum and diff.
	# Because ee.vadds/vsubs are 3-operand (destination need not alias
	# either source), the sum can be written to the still-free scratch
	# register q7 while both original operands stay intact for the diff -
	# no extra register needed beyond the one spare already secured above.

	# pair (tmp0=q0, tmp7=q6) -> rows 0, 7
	ee.vadds.s16	q7,q0,q6   # row0 = tmp0+tmp7
	ee.vsubs.s16	q6,q0,q6   # row7 = tmp0-tmp7
	addi	a5,a2,0
	ee.vst.128.ip	q7,a5,0
	addi	a5,a2,112
	ee.vst.128.ip	q6,a5,0
	# q0,q6,q7 now free

	# pair (tmp1=q4, tmp6=q2) -> rows 1, 6
	ee.vadds.s16	q0,q4,q2   # row1 = tmp1+tmp6
	ee.vsubs.s16	q2,q4,q2   # row6 = tmp1-tmp6
	addi	a5,a2,16
	ee.vst.128.ip	q0,a5,0
	addi	a5,a2,96
	ee.vst.128.ip	q2,a5,0
	# q0,q2,q4 now free (q1=tmp4, q3=tmp5, q6=tmp7 already stored/free as noted)

	# pair (tmp2=q5, tmp5=q3) -> rows 2, 5
	ee.vadds.s16	q0,q5,q3   # row2 = tmp2+tmp5
	ee.vsubs.s16	q3,q5,q3   # row5 = tmp2-tmp5
	addi	a5,a2,32
	ee.vst.128.ip	q0,a5,0
	addi	a5,a2,80
	ee.vst.128.ip	q3,a5,0

	# pair (tmp3, tmp4=q1) -> rows 3, 4  (tmp3 reloaded now - a9 already
	# points at its spill slot from the partial reload above)
	ee.vld.128.ip	q0,a9,0    # q0 = tmp3
	ee.vadds.s16	q2,q0,q1   # row4 = tmp3+tmp4
	ee.vsubs.s16	q1,q0,q1   # row3 = tmp3-tmp4
	addi	a5,a2,64
	ee.vst.128.ip	q2,a5,0
	addi	a5,a2,48
	ee.vst.128.ip	q1,a5,0

	retw.n
#endif // dsps_fft2r_sc16_aes3_enabled
#endif // ESP32
