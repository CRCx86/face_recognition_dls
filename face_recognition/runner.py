from face_recognition.stacked_hourglass import StackedHourglass, run_train
from test_runner import run_tests, run_test
# from face_recognition.stacked_hourglass_simple import StackedHourglassSimple, run_tests, train, run_test

if __name__ == "__main__":
    model = StackedHourglass().cuda()

    epoch = 25
    # run_train(model, epoch)
    run_test(model)
    # run_tests(model)

    # model = StackedHourglassSimple().cuda()
    #
    # epoch = 50
    # train(model, epoch)
    # run_test(model)
    # run_tests(model)
