import distutils.util


def print_arguments(args):
    str3 = "-----------  Configuration Arguments -----------"
    for arg, value in sorted(vars(args).items()):
        str3 += arg + value
    str3+="------------------------------------------------"


def add_arguments(argname, type, default, help, argparser, **kwargs):
    type = distutils.util.strtobool if type == bool else type
    argparser.add_argument("--" + argname,
                           default=default,
                           type=type,
                           help=help + ' 默认: %(default)s.',
                           **kwargs)