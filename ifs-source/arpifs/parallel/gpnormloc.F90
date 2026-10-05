SUBROUTINE GPNORMLOC(YDGEOMETRY,PGP,KFIELDS,PNORMS,PGPD,PNORMSD)

!**** *gpnormloc* - Routine to calculate grid point norm by 
!                   particular sumations in GP space.

! Note: This is an alternative version of computing global norms.
!       It is done with great care for the numerical accuracy
!       using the method of Knuth (1969). Even though, the summation
!       is exclusively done by double precision ensuring the 
!       norms for sparse fields (like water tracers) are being
!       consistently conserved.

!     Purpose.
!     --------
! Routine to calculate grid point norms.

!**   Interface.
!     ----------
!        *CALL* *GPNORMLOC(...) *

!        Explicit arguments :
!        --------------------

!        INPUT:
!        ------
!        PGP      : if nproc is > 1 then PGP is the distributed field
!        KFIELDS  : number of levels/fields

!        OUTPUT:
!        -------
!        PNORM    : Grid-point norm

!        Implicit arguments :
!        --------------------

!     Method.
!     -------
!        See documentation

!     Externals.
!     ----------

!     Reference.
!     ----------
!        ECMWF Research Department documentation of the IFS

!     Author.
!     -------
!      F. Vana    *ECMWF*
!      Original : 14-Aug-2026

!     Modifications.
!     --------------

!     ------------------------------------------------------------------

USE GEOMETRY_MOD , ONLY : GEOMETRY
USE PARKIND1     , ONLY : JPIM,  JPRB,  JPRD
USE YOMHOOK      , ONLY : LHOOK, DR_HOOK, JPHOOK
USE ORDER_INDEPENDENT_SUMMATION_MOD, ONLY : ORDER_INDEP_GLOBAL_SUM_DP

!     ------------------------------------------------------------------

IMPLICIT NONE

TYPE(GEOMETRY)    ,INTENT(IN)    :: YDGEOMETRY
REAL(KIND=JPRB)   ,INTENT(IN)    :: PGP(YDGEOMETRY%YRDIM%NPROMA,KFIELDS, &
  &  YDGEOMETRY%YRDIM%NGPBLKS) 
INTEGER(KIND=JPIM),INTENT(IN)    :: KFIELDS 
REAL(KIND=JPRB), INTENT(OUT)     :: PNORMS(3,KFIELDS) 
REAL(KIND=JPRD),OPTIONAL,  INTENT(IN) :: PGPD(YDGEOMETRY%YRDIM%NPROMA,KFIELDS, &
  &  YDGEOMETRY%YRDIM%NGPBLKS) 
REAL(KIND=JPRD), OPTIONAL, INTENT(OUT):: PNORMSD(3,KFIELDS) 


!     ------------------------------------------------------------------

INTEGER(KIND=JPIM) :: JL,JKGLO,IST,IEND,IBL,IGLB

REAL(KIND=JPRD),ALLOCATABLE :: ZGP(:)

REAL(KIND=JPRD) :: ZNORM(KFIELDS)

REAL(KIND=JPHOOK) :: ZHOOK_HANDLE

!     ------------------------------------------------------------------

IF (LHOOK) CALL DR_HOOK('GPNORMLOC',0,ZHOOK_HANDLE)
ASSOCIATE(YDDIM=>YDGEOMETRY%YRDIM, YDGEM=>YDGEOMETRY%YRGEM, &
 & YDGSGEOM=>YDGEOMETRY%YRGSGEOM)
ASSOCIATE(NGPBLKS=>YDDIM%NGPBLKS, NPROMA=>YDDIM%NPROMA, &
 & NGPTOT=>YDGEM%NGPTOT,NGPTOTG=>YDGEM%NGPTOTG)
!     ------------------------------------------------------------------

ALLOCATE(ZGP(NGPTOT))

CALL GSTATS(1427,0)

DO JL=1,KFIELDS 

 IF (PRESENT(PGPD)) THEN

!$OMP PARALLEL DO SCHEDULE(STATIC) &
!$OMP& PRIVATE(JKGLO,IST,IEND,IBL,IGLB)
    DO JKGLO=1,NGPTOT,NPROMA
      IST=1
      IEND=MIN(NPROMA,NGPTOT-JKGLO+1)
      IBL=(JKGLO-1)/NPROMA+1
      IGLB=(IBL-1)*NPROMA

      ZGP(IGLB+IST:IGLB+IEND)=PGPD(IST:IEND,JL,IBL) &
        & *YDGSGEOM(IBL)%GAW(IST:IEND)

    ENDDO
!$OMP END PARALLEL DO

 ELSE

!$OMP PARALLEL DO SCHEDULE(STATIC) &
!$OMP& PRIVATE(JKGLO,IST,IEND,IBL,IGLB)
    DO JKGLO=1,NGPTOT,NPROMA
      IST=1
      IEND=MIN(NPROMA,NGPTOT-JKGLO+1)
      IBL=(JKGLO-1)/NPROMA+1
      IGLB=(IBL-1)*NPROMA

      ZGP(IGLB+IST:IGLB+IEND)=REAL(PGP(IST:IEND,JL,IBL) &
        & *YDGSGEOM(IBL)%GAW(IST:IEND),JPRD)

    ENDDO
!$OMP END PARALLEL DO

  ENDIF

  ZNORM(JL) = ORDER_INDEP_GLOBAL_SUM_DP(ZGP,KNG=NGPTOT)

ENDDO

CALL GSTATS(1427,1)

PNORMS(1,1:KFIELDS)=REAL(ZNORM(1:KFIELDS),JPRB) !/REAL(NGPTOTG,JPRD)

IF (PRESENT(PNORMSD)) PNORMSD(1,1:KFIELDS)=ZNORM(1:KFIELDS)

DEALLOCATE(ZGP)

!     ------------------------------------------------------------------
END ASSOCIATE
END ASSOCIATE
IF (LHOOK) CALL DR_HOOK('GPNORMLOC',1,ZHOOK_HANDLE)
END SUBROUTINE GPNORMLOC
