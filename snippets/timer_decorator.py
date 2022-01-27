from time import sleep, time


def timer(func):
    def func_wrap(*args, **kwargs):
        tstart = time()
        result = func(*args, **kwargs)
        tend = time()
        print(f"Function {func.__name__!r} executed in {(tend-tstart):.3f}s")
        return result

    return func_wrap


@timer
def looping(n):
    for i in range(n):
        sleep(1)


looping(10)
