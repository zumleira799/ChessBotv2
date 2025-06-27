.global writeF
.global readF
.bss
stupidBuffer: .skip 8
statBuffer: .skip 48
filesize: .skip 200

.data
fileName: .asciz "../trainingData/NNbinaries"

.text

writeF:
    pushq %rbp
    movq %rsp, %rbp
    
    pushq %rsi
    xorq %rsi, %rsi
    movl (%rsp), %esi
    addq $8, %rsp

    movq %rsi, %rax
    imulq %rdx
    movq %rax, %rsi

    pushq %rdi
    pushq %rsi

    pushq %rcx
    movq $87, %rax
    movq %rcx, %rdi
    syscall
    popq %rcx

    popq %rsi
    popq %rdi

    pushq %r10

    pushq %rdi
    pushq %rsi

    movq $2, %rax
    movq %rcx, %rdi
    movq $102, %rsi
    movq $0644, %rdx
    syscall

    popq %rsi
    popq %rdi

    pushq %rdi
    pushq %rsi

    movq %rax, %r10

    movq $1, %rax
    movq %r10, %rdi
    movq %rsi, %rdx
    movq 8(%rsp), %rsi
    syscall

    popq %rsi
    popq %rdi

    movq $3, %rax
    movq %r10, %rdi
    syscall

    popq %r10
    movq %rbp, %rsp
    popq %rbp
    ret


readF:
    pushq %r12
    pushq %r13
    pushq %r14

    pushq %rbp
    movq %rsp, %rbp
    movq %rsi, %r12

    movq $0, (%rdx)

    movq %rdx, %r13
    xorq %r14, %r14

    pushq %rdi

    movq $2, %rax
    movq $102, %rsi
    movq $0644, %rdx
    syscall

    pushq %rax
    pushq %r10
    xorq %r10, %r10
    subq %r12, %r10

    decq %r14

    movq $5, %rax
    movq 8(%rsp), %rdi
    movq $statBuffer, %rsi
    syscall

    movq filesize, %rax
    movq %rax, %r10
    xorq %rdx, %rdx
    divq %r12

    movq %rax, (%r13)

    pushq %r10

    movq $9, %rax
    movq $0, %rdi
    movq %r10, %rsi
    movq $0x3, %rdx
    movq $0x22, %r10
    movq $-1, %r8
	movq $0, %r9
    syscall

    popq %r10
    
    pushq %rax

    movq $2, %rax
    movq 24(%rsp), %rdi
    movq $102, %rsi
    movq $0644, %rdx
    syscall

    movq %rax, %rdi
    xorq %rax, %rax
    movq (%rsp), %rsi
    movq %r10, %rdx
    syscall

    movq $3, %rax
    movq 16(%rsp), %rdi
    syscall

    popq %rax
    popq %r10

    movq %rbp, %rsp
    popq %rbp
    popq %r14
    popq %r13
    popq %r12
    ret










