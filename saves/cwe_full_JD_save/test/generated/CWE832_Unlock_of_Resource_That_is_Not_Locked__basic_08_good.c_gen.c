int staticReturnsTrue ( ) { return 1 ; } int stdThreadLockDestroy ( ) { return 0 ; } void name_0 ( ) { if (  staticReturnsTrue  ( ) ) { { static stdThreadLock  name_1  =  NULL  ; if ( !   stdThreadLockCreate  ( &  name_1  )  ) {  exit  ( 1 ) ; }  stdThreadLockAcquire  ( name_1 ) ;  stdThreadLockRelease  ( name_1 ) ;  name_0  ( name_1 ) ; } } } int main ( int argc char * argv [ ] ) {  srand  ( ( unsigned )   time  ( NULL )  ) ;  stdThreadLockDestroy  ( ) ; return 0 ; } 