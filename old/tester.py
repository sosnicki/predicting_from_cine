import joblib
from multiprocessing import Manager


def worker_function(queue, data):
    for item in data:
        queue.put(item)


def main():
    # Create a multiprocessing Manager
    manager = Manager()

    # Create a shared Queue
    shared_queue = manager.Queue()

    # Your data
    data = range(10)

    # Run worker function in parallel
    joblib.Parallel(n_jobs=-1, backend="loky")(
        joblib.delayed(worker_function)(shared_queue, data) for _ in range(5)
    )

    # Process the shared queue
    while not shared_queue.empty():
        print(shared_queue.get())


if __name__ == "__main__":
    main()